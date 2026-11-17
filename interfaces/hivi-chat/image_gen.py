"""IMG-1: local image generation through stable-diffusion.cpp (``sd-cli``).

Why a module of its own:

* **Discovery, not environment variables.** The desktop shortcut starts the server
  with a clean environment, so paths set in a PowerShell session are not seen.
  Paths come from ``configs/image.yaml`` (optional) and otherwise from a search of
  ``Models/Image`` and the inference folder. ``status()`` says what was found.
* **GPU handoff.** An 8 GB card cannot hold the chat model and the image model at
  once: the server unloads the chat model before a render and reloads it after.
* **Progress.** ``sd-cli`` prints its sampling steps; a job parses them so the UI
  can show a live canvas instead of a blank wait, and a job can be cancelled.
"""
from __future__ import annotations

import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from hivi_paths import paths

SIZES = (512, 640, 768, 896, 1024, 1152, 1280)
MAX_STEPS = 50
DEFAULT_STEPS = 4
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*-\s*([\d.]+)\s*(s/it|it/s)")
_PCT_RE = re.compile(r"\|[=> ]*\|\s*(\d+)\s*/\s*(\d+)")
_TEXT_ENCODER_HINT = re.compile(r"qwen[\d.]*[-_]?vl|text[-_]?encoder|\bt5|clip|mmproj", re.IGNORECASE)
_WEIGHT_SUFFIXES = (".gguf", ".safetensors")
_VAE_HINT = re.compile(r"vae", re.IGNORECASE)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# --------------------------------------------------------------------------- #
# Configuration and discovery
# --------------------------------------------------------------------------- #

def config_file() -> Path:
    return paths.repo_root() / "configs" / "image.yaml"


def load_config() -> dict[str, Any]:
    try:
        data = yaml.safe_load(config_file().read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        data = {}
    return data if isinstance(data, dict) else {}


def image_root() -> Path:
    cfg = load_config()
    custom = str(cfg.get("root") or os.environ.get("HIVI_IMAGE_ROOT") or "").strip()
    return Path(custom).expanduser() if custom else paths.models_root() / "Image"


def _first(paths_: list[Path]) -> Path | None:
    return paths_[0] if paths_ else None


def _find_cli(root: Path) -> Path | None:
    cfg = load_config()
    for cand in (cfg.get("cli"), os.environ.get("HIVI_SD_CLI")):
        if cand and Path(cand).is_file():
            return Path(cand)
    names = ("sd-cli.exe", "sd-cli", "sd.exe", "sd")
    for base in (root, paths.inference_root()):
        if not base.is_dir():
            continue
        for name in names:
            hits = sorted(base.rglob(name), key=lambda p: (len(p.parts), str(p)))
            hits = [h for h in hits if h.is_file()]
            if hits:
                return hits[0]
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _find_model(root: Path) -> Path | None:
    cfg = load_config()
    for cand in (cfg.get("model"), os.environ.get("HIVI_IMAGE_MODEL")):
        if cand and Path(cand).is_file():
            return Path(cand)
    if not root.is_dir():
        return None
    ggufs = [p for p in root.rglob("*.gguf")
             if "text_encoders" not in p.parts and not _TEXT_ENCODER_HINT.search(p.name)
             and not _VAE_HINT.search(p.name)]
    # Prefer a Q4_K_M build of a Qwen Image model, then any Qwen Image, then anything.
    for pattern in (r"qwen[-_]?image.*q4_k_m", r"qwen[-_]?image", r"flux", r"."):
        hits = sorted(p for p in ggufs if re.search(pattern, p.name, re.IGNORECASE))
        if hits:
            return hits[0]
    return None


def _find_text_encoder(root: Path) -> Path | None:
    cfg = load_config()
    for cand in (cfg.get("text_encoder"), os.environ.get("HIVI_IMAGE_LLM")):
        if cand and Path(cand).is_file():
            return Path(cand)
    if not root.is_dir():
        return None
    def weights(folder: Path) -> list[Path]:
        return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES
                      and not p.name.startswith("mmproj") and not _VAE_HINT.search(p.name))

    # Anything placed in the text_encoders folder is the text encoder.
    for folder in (root / "text_encoders", root / "text_encoder", root / "llm"):
        if folder.is_dir():
            files = weights(folder)
            for pattern in (r"q4_k_m", r"."):
                hits = [p for p in files if re.search(pattern, p.name, re.IGNORECASE)]
                if hits:
                    return hits[0]
    if root.is_dir():
        for pattern in (r"qwen[\d.]*[-_]?vl.*q4_k_m", r"qwen[\d.]*[-_]?vl", r"text[-_]?encoder"):
            hits = [p for p in weights(root) if re.search(pattern, p.name, re.IGNORECASE)]
            if hits:
                return hits[0]
    return None


def _find_vae(root: Path) -> Path | None:
    cfg = load_config()
    for cand in (cfg.get("vae"), os.environ.get("HIVI_IMAGE_VAE")):
        if cand and Path(cand).is_file():
            return Path(cand)
    if not root.is_dir():
        return None
    hits = sorted(p for p in root.rglob("*") if p.is_file() and _VAE_HINT.search(p.name)
                  and p.suffix.lower() in (".safetensors", ".gguf", ".pt", ".ckpt"))
    return _first(hits)


@dataclass
class Setup:
    cli: Path | None
    model: Path | None
    text_encoder: Path | None
    vae: Path | None
    root: Path
    extra_args: list[str]
    needs_text_encoder: bool
    unload_chat: bool

    @property
    def ready(self) -> bool:
        return bool(self.cli and self.model and (self.text_encoder or not self.needs_text_encoder))

    def missing(self) -> list[str]:
        out = []
        if not self.cli:
            out.append("sd-cli.exe (stable-diffusion.cpp) not found under Models\\Image or the inference folder")
        if not self.model:
            out.append("no image model (*.gguf) found under Models\\Image")
        if self.needs_text_encoder and not self.text_encoder:
            out.append("Qwen Image needs its Qwen-VL text encoder: put the file (GGUF or safetensors) in "
                       "Models\\Image\\text_encoders, or set text_encoder: in configs\\image.yaml")
        return out


def discover() -> Setup:
    cfg = load_config()
    root = image_root()
    model = _find_model(root)
    needs_te = bool(model and re.search(r"qwen[-_]?image", model.name, re.IGNORECASE))
    if "needs_text_encoder" in cfg:
        needs_te = bool(cfg["needs_text_encoder"])
    extra = cfg.get("extra_args") or os.environ.get("HIVI_IMAGE_ARGS", "")
    extra_args = extra.split() if isinstance(extra, str) else [str(a) for a in extra]
    unload = cfg.get("unload_chat_model")
    if unload is None:
        unload = os.environ.get("HIVI_IMAGE_UNLOAD_CHAT", "1") not in ("0", "false", "no")
    return Setup(cli=_find_cli(root), model=model, text_encoder=_find_text_encoder(root),
                 vae=_find_vae(root), root=root, extra_args=extra_args,
                 needs_text_encoder=needs_te, unload_chat=bool(unload))


def status() -> dict[str, Any]:
    s = discover()
    return {
        "ready": s.ready,
        "missing": s.missing(),
        "cli": str(s.cli) if s.cli else None,
        "model": s.model.name if s.model else None,
        "model_path": str(s.model) if s.model else None,
        "text_encoder": s.text_encoder.name if s.text_encoder else None,
        "vae": s.vae.name if s.vae else None,
        "root": str(s.root),
        "config": str(config_file()),
        "unload_chat_model": s.unload_chat,
        "sizes": list(SIZES),
        "default_steps": DEFAULT_STEPS,
    }


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #

def images_dir() -> Path:
    d = paths.data_root() / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Job:
    id: str
    prompt: str
    width: int
    height: int
    steps: int
    seed: int | None
    state: str = "queued"           # queued | unloading | loading | rendering | decoding | reloading | done | failed | cancelled
    stage: str = "Queued"
    step: int = 0
    total_steps: int = 0
    percent: float = 0.0
    message: str = ""
    error: str | None = None
    filename: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    chat_model_id: str | None = None
    log_tail: list[str] = field(default_factory=list)
    _proc: subprocess.Popen | None = field(default=None, repr=False)
    _cancel: bool = field(default=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        elapsed = (self.finished_at or time.time()) - (self.started_at or self.created_at)
        return {
            "id": self.id, "prompt": self.prompt, "width": self.width, "height": self.height,
            "steps": self.steps, "seed": self.seed, "state": self.state, "stage": self.stage,
            "step": self.step, "total_steps": self.total_steps, "percent": round(self.percent, 1),
            "message": self.message, "error": self.error,
            "image_url": f"/api/images/{self.filename}" if self.filename and self.state == "done" else None,
            "elapsed_s": round(max(0.0, elapsed), 1),
            "chat_model_id": self.chat_model_id,
            "done": self.state in ("done", "failed", "cancelled"),
        }

    def _set(self, state: str, stage: str, percent: float | None = None, message: str = "") -> None:
        self.state = state
        self.stage = stage
        if percent is not None:
            self.percent = percent
        self.message = message

    def cancel(self) -> None:
        self._cancel = True
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()   # one render at a time
_KEEP_JOBS = 50


def get_job(job_id: str) -> Job | None:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def _remember(job: Job) -> None:
    with _JOBS_LOCK:
        _JOBS[job.id] = job
        if len(_JOBS) > _KEEP_JOBS:
            for old in sorted(_JOBS.values(), key=lambda j: j.created_at)[: len(_JOBS) - _KEEP_JOBS]:
                if old.state in ("done", "failed", "cancelled"):
                    _JOBS.pop(old.id, None)


def _parse_progress(line: str, job: Job) -> None:
    m = _STEP_RE.search(line) or _PCT_RE.search(line)
    if not m:
        return
    step, total = int(m.group(1)), int(m.group(2))
    if total <= 0:
        return
    job.step, job.total_steps = step, total
    # Rendering spans 20–85 % of the whole job; the rest is model load and VAE decode.
    job.percent = 20 + 65 * min(1.0, step / total)
    job.stage = f"Rendering step {step} of {total}"
    if m.lastindex and m.lastindex >= 4:
        rate = m.group(3)
        job.message = f"{rate} {m.group(4)}"


def build_command(setup: Setup, job: Job, output: Path) -> list[str]:
    cmd = [str(setup.cli), "--diffusion-model", str(setup.model), "--prompt", job.prompt,
           "--output", str(output), "--width", str(job.width), "--height", str(job.height),
           "--steps", str(job.steps)]
    if setup.text_encoder:
        cmd += ["--llm", str(setup.text_encoder)]
    if setup.vae:
        cmd += ["--vae", str(setup.vae)]
    if job.seed is not None:
        cmd += ["--seed", str(job.seed)]
    cmd += setup.extra_args
    return cmd


def run_job(job: Job, *, unload_chat=None, reload_chat=None, timeout_s: float | None = None) -> None:
    """Blocking; call from a worker thread. ``unload_chat()`` returns the model id to
    reload later (or None); ``reload_chat(model_id)`` restores it."""
    timeout_s = timeout_s or float(os.environ.get("HIVI_IMAGE_TIMEOUT_S", "900"))
    with _RUN_LOCK:
        job.started_at = time.time()
        if job._cancel:
            job._set("cancelled", "Cancelled", 0)
            job.finished_at = time.time()
            return
        setup = discover()
        if not setup.ready:
            job.error = "; ".join(setup.missing())
            job._set("failed", "Not set up", 0)
            job.finished_at = time.time()
            return
        reload_id = None
        try:
            if setup.unload_chat and unload_chat is not None:
                job._set("unloading", "Freeing the GPU (unloading the chat model)", 5)
                reload_id = unload_chat()
                job.chat_model_id = reload_id
            if job._cancel:
                raise _Cancelled()
            job._set("loading", "Loading the image model", 12)
            output = images_dir() / f"{secrets.token_urlsafe(18).replace('-', '_')}.png"
            cmd = build_command(setup, job, output)
            proc = subprocess.Popen(
                cmd, cwd=str(setup.cli.parent), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=_NO_WINDOW,
            )
            job._proc = proc
            deadline = time.time() + timeout_s
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\r\n")
                for piece in re.split(r"\r", line):
                    piece = piece.strip()
                    if not piece:
                        continue
                    job.log_tail.append(piece[-300:])
                    if len(job.log_tail) > 40:
                        job.log_tail.pop(0)
                    low = piece.lower()
                    if "sampling" in low or _STEP_RE.search(piece) or _PCT_RE.search(piece):
                        if job.state != "rendering":
                            job._set("rendering", "Rendering", 20)
                        _parse_progress(piece, job)
                    elif "decod" in low and job.state == "rendering":
                        job._set("decoding", "Decoding the image", 88)
                    elif "loading" in low and job.state == "loading":
                        job.message = piece[:120]
                if time.time() > deadline:
                    proc.kill()
                    raise TimeoutError(f"image generation exceeded {int(timeout_s)} s")
                if job._cancel:
                    raise _Cancelled()
            code = proc.wait()
            job._proc = None
            if job._cancel:
                raise _Cancelled()
            if code != 0 or not output.is_file():
                tail = "\n".join(job.log_tail[-8:]) or "image generator failed"
                raise RuntimeError(tail[-1200:])
            job.filename = output.name
            _write_sidecar(output, job, setup)
            job._set("done", "Done", 100)
        except _Cancelled:
            job._set("cancelled", "Cancelled", 0)
        except (RuntimeError, TimeoutError, OSError) as e:
            job.error = str(e)
            job._set("failed", "Failed", 0)
        finally:
            job._proc = None
            if reload_id and reload_chat is not None:
                try:
                    job.stage = "Reloading the chat model"
                    job.state = "reloading" if job.state == "done" else job.state
                    reload_chat(reload_id)
                except Exception as e:  # noqa: BLE001 - the image is already made
                    job.message = f"chat model did not reload: {e}"
                if job.state == "reloading":
                    job._set("done", "Done", 100)
            job.finished_at = time.time()


class _Cancelled(Exception):
    pass


def _write_sidecar(output: Path, job: Job, setup: Setup) -> None:
    try:
        meta = {"prompt": job.prompt, "width": job.width, "height": job.height, "steps": job.steps,
                "seed": job.seed, "model": setup.model.name if setup.model else None,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        output.with_suffix(".json").write_text(yaml.safe_dump(meta, allow_unicode=True), encoding="utf-8")
    except OSError:
        pass


def new_job(prompt: str, *, width: int, height: int, steps: int, seed: int | None) -> Job:
    job = Job(id=uuid.uuid4().hex[:16], prompt=prompt, width=width, height=height, steps=steps, seed=seed)
    _remember(job)
    return job


def list_images(limit: int = 60) -> list[dict[str, Any]]:
    out = []
    for p in sorted(images_dir().glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        meta = {}
        side = p.with_suffix(".json")
        if side.is_file():
            try:
                meta = yaml.safe_load(side.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                meta = {}
        out.append({"image_url": f"/api/images/{p.name}", "filename": p.name,
                    "prompt": meta.get("prompt", ""), "seed": meta.get("seed"),
                    "created": meta.get("created"), "size_kb": round(p.stat().st_size / 1024)})
    return out
