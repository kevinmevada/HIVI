"""IMG-1: local image generation through stable-diffusion.cpp (``sd-cli``).

Why a module of its own:

* **Discovery, not environment variables.** The desktop shortcut starts the server
  with a clean environment, so paths set in a PowerShell session are not seen.
  Paths come from ``configs/image.yaml`` (optional) and otherwise from a search of
  ``Models/Image`` and the inference folder. ``status()`` says what was found.
* **GPU handoff.** An 8 GB card cannot hold the chat model and the image model at
  once. The server parks the chat model for the render (holding the switch lock so
  nothing else can load onto the GPU meanwhile) and reloads it afterwards; see
  ``server._run_image_job``. This module only renders.
* **Right settings per model.** Qwen-Image 2.1 Turbo wants its 8-step sigma preset
  at CFG 1; the base model 20 steps at CFG 6; both 1024 px. ``profile_for`` picks
  them, ``build_command`` adds speed flags the installed ``sd-cli`` supports
  (``--diffusion-fa``, ``--offload-to-cpu``, ``--vae-tiling``), and anything in
  ``extra_args`` wins.
* **Progress.** ``sd-cli`` prints its sampling steps; a job parses them so the UI
  can show a live canvas instead of a blank wait, and a job can be cancelled.
"""
from __future__ import annotations

import json
import os
import random
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from hivi_paths import paths

MIN_SIDE, MAX_SIDE = 512, 1536
MAX_PIXELS = 1536 * 1024          # bigger than this does not fit an 8 GB card sensibly
MAX_STEPS = 60
# Aspect presets (Qwen-Image's own sizes, all multiples of 32)
ASPECTS = {
    "1:1": (1024, 1024),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
    "16:9": (1536, 864),
    "9:16": (864, 1536),
}
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*-\s*([\d.]+)\s*(s/it|it/s)")
_PCT_RE = re.compile(r"\|[=> ]*\|\s*(\d+)\s*/\s*(\d+)")
_TEXT_ENCODER_HINT = re.compile(r"qwen[\d.]*[-_]?vl|text[-_]?encoder|\bt5|clip|mmproj", re.IGNORECASE)
_WEIGHT_SUFFIXES = (".gguf", ".safetensors")
_VAE_HINT = re.compile(r"vae", re.IGNORECASE)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_WIN_NATIVE = {".exe", ".com", ".bat", ".cmd"}


def cli_argv(cli: Path, *args: str) -> list[str]:
    """Build argv for sd-cli / sd-server. Windows cannot run shebang scripts directly."""
    argv = [str(cli), *args]
    if os.name != "nt" or cli.suffix.lower() in _WIN_NATIVE:
        return argv
    try:
        head = cli.read_bytes()[:160]
    except OSError:
        return argv
    line = head.split(b"\n", 1)[0].lower()
    if head.startswith(b"#!") and b"python" in line:
        return [sys.executable, str(cli), *args]
    return argv


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


@dataclass(frozen=True)
class Profile:
    name: str
    steps: int
    cfg_scale: float
    sampler: str
    sigmas: str | None
    size: tuple[int, int]


def profile_for(model: Path | None) -> Profile:
    """Recommended settings for the model family (configs/image.yaml can override)."""
    n = model.name.lower() if model else ""
    qwen21 = re.search(r"qwen[-_]?image[-_]?2[._]?1", n)
    if qwen21 and "turbo" in n:
        p = Profile("Qwen-Image 2.1 Turbo", 8, 1.0, "euler", "qwen-image-2.1-turbo", (1024, 1024))
    elif qwen21:
        p = Profile("Qwen-Image 2.1", 20, 6.0, "euler", None, (1024, 1024))
    elif re.search(r"qwen[-_]?image", n):
        p = Profile("Qwen-Image", 20, 2.5, "euler", None, (1024, 1024))
    elif re.search(r"turbo|schnell|lightning", n):
        p = Profile("Few-step model", 8, 1.0, "euler", None, (1024, 1024))
    else:
        p = Profile("Image model", 20, 5.0, "euler", None, (768, 768))
    cfg = load_config()
    try:
        return Profile(
            p.name,
            int(cfg.get("steps") or p.steps),
            float(cfg.get("cfg_scale") or p.cfg_scale),
            str(cfg.get("sampler") or p.sampler),
            cfg.get("sigmas", p.sigmas) or None,
            (int(cfg.get("width") or p.size[0]), int(cfg.get("height") or p.size[1])),
        )
    except (TypeError, ValueError):
        return p


_HELP_CACHE: dict[str, set[str]] = {}


def cli_flags(cli: Path | None) -> set[str]:
    """Flags this sd-cli build accepts (from ``--help``), so new flags are never
    passed to an old binary. Empty when the help text could not be read."""
    if cli is None:
        return set()
    try:
        key = f"{cli}:{cli.stat().st_mtime_ns}"
    except OSError:
        return set()
    if key in _HELP_CACHE:
        return _HELP_CACHE[key]
    try:
        out = subprocess.run(cli_argv(cli, "--help"), capture_output=True, text=True, timeout=20,
                             check=False, cwd=str(cli.parent), creationflags=_NO_WINDOW,
                             encoding="utf-8", errors="replace")
        text = (out.stdout or "") + (out.stderr or "")
    except (OSError, subprocess.SubprocessError):
        text = ""
    flags = set(re.findall(r"(?<![\w-])--[a-z0-9][a-z0-9_-]*", text))
    _HELP_CACHE[key] = flags
    return flags


def speed_flags(supported: set[str], *, low_vram: bool) -> list[list[str]]:
    """Speed / memory flags for this sd-cli or sd-server build.

    Newer builds auto-fit weights to free VRAM (``--auto-fit``, on by default) and
    pick flash attention themselves (``--flash-attn auto``); forcing
    ``--offload-to-cpu`` there makes every picture copy all weights across PCIe.
    So offload and VAE tiling are used only on older builds, or after a render ran
    out of memory (``low_vram``)."""
    cfg = load_config()
    out: list[list[str]] = []
    if not supported:
        return out
    if cfg.get("flash_attention", True) and "--flash-attn" not in supported and "--diffusion-fa" in supported:
        out.append(["--diffusion-fa"])
    if cfg.get("mmap", True) and "--mmap" in supported:
        out.append(["--mmap"])          # reloads come from the OS file cache
    mode = str(cfg.get("low_vram", "auto")).lower()
    forced = mode in ("true", "yes", "on", "1")
    off = mode in ("false", "no", "off", "0")
    if not off and (forced or low_vram or "--auto-fit" not in supported):
        if "--offload-to-cpu" in supported:
            out.append(["--offload-to-cpu"])
        if "--vae-tiling" in supported:
            out.append(["--vae-tiling"])
    return out


_STATE_FILE_NAME = "image_state.json"


def _state_file() -> Path:
    return paths.data_root() / "run" / _STATE_FILE_NAME


def learned_low_vram() -> bool:
    try:
        return bool(json.loads(_state_file().read_text(encoding="utf-8")).get("low_vram"))
    except (OSError, ValueError):
        return False


def remember_low_vram(model: Path | None) -> None:
    try:
        f = _state_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"low_vram": True, "model": str(model) if model else None}), encoding="utf-8")
    except OSError:
        pass


def _is_oom(text: str) -> bool:
    return bool(re.search(r"out of memory|cudaMalloc failed|failed to allocate|OOM", text or "", re.IGNORECASE))


def valid_size(width: int, height: int) -> bool:
    return (MIN_SIDE <= width <= MAX_SIDE and MIN_SIDE <= height <= MAX_SIDE
            and width % 32 == 0 and height % 32 == 0 and width * height <= MAX_PIXELS)


def status() -> dict[str, Any]:
    s = discover()
    prof = profile_for(s.model)
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
        "profile": prof.name,
        "steps": prof.steps,
        "cfg_scale": prof.cfg_scale,
        "default_width": prof.size[0],
        "default_height": prof.size[1],
        "aspects": {k: list(v) for k, v in ASPECTS.items()},
        "enhance_prompt": enhance_enabled(),
        "reload_delay_s": reload_delay_s(),
    }


def enhance_enabled() -> bool:
    v = load_config().get("enhance_prompt")
    return True if v is None else bool(v)


def reload_delay_s() -> float:
    """Seconds the chat model stays parked after a render, so a quick Regenerate
    does not pay for a reload and a second unload."""
    try:
        return max(0.0, float(load_config().get("reload_delay_s", 20)))
    except (TypeError, ValueError):
        return 20.0


# --------------------------------------------------------------------------- #
# Prompt enhancement (runs on the chat model *before* it is parked)
# --------------------------------------------------------------------------- #

ENHANCE_SYSTEM = (
    "You write prompts for a text-to-image model. Rewrite the user's request as ONE "
    "detailed image prompt in English: the subject and what it is doing, setting, "
    "composition and framing, lighting, colour palette, mood, and medium or style "
    "(photo with lens and film look, illustration, 3D render, poster, etc.). Keep every "
    "specific detail the user gave and never contradict it. If the user wants words in "
    "the image, keep them exactly, in double quotes. 50 to 90 words. Output only the "
    "prompt: no preamble, no quotes around it, no lists."
)


def needs_enhancing(prompt: str) -> bool:
    words = len(prompt.split())
    return words < 40 and not re.search(r"\b(exact(ly)?|verbatim|as written|no changes)\b", prompt, re.IGNORECASE)


_ASK = re.compile(
    r"^(?:please\s+|pls\s+|hivi[,!]?\s+|hey\s+)*(?:can you\s+|could you\s+)?"
    r"(?:get|give|make|create|generate|draw|paint|render|show)\s+(?:me\s+)?(?:an?\s+|the\s+|some\s+)?"
    r"(?:image|picture|photo|pic|illustration|artwork|drawing|painting|render)s?\s+(?:of\s+)?",
    re.IGNORECASE,
)
_RES_WORDS = re.compile(r"\b(?:4k|8k|uhd|hd|full hd|high[- ]res(?:olution)?|ultra[- ]?realistic)\b", re.IGNORECASE)
_STYLE_WORDS = re.compile(
    r"\b(?:anime|cartoon|illustration|illustrated|drawing|sketch|painting|watercolor|watercolour|oil|"
    r"pixel|3d|render|low[- ]poly|vector|logo|icon|poster|comic|manga|digital art|concept art|clay)\b",
    re.IGNORECASE,
)
_PHOTO_CUES = ("photograph, natural daylight, true-to-life colours, fine natural detail, sharp focus, "
               "realistic textures, shot on a full-frame camera with a 35mm lens, subtle depth of field")


def polish_prompt(prompt: str) -> str:
    """No chat model available: still turn a request into a usable image prompt.

    Strips the request wording ("get me an image of"), drops resolution buzzwords
    the model ignores, and for anything that is not a named art style adds the
    photographic cues Qwen-Image needs to look natural instead of plasticky."""
    t = _ASK.sub("", prompt.strip())
    t = re.sub(r"^(?:please\s+|can you\s+|could you\s+)*(?:get|give|make|create|generate|draw|paint|render)"
               r"\s+(?:me\s+|us\s+)?", "", t, flags=re.IGNORECASE)
    wants_real = bool(re.search(r"\b(?:realistic|real|photo(?:graph)?|photoreal\w*|natural)\b", t, re.IGNORECASE))
    t = _RES_WORDS.sub("", t)
    t = re.sub(r"\b(?:a|an)?\s*realistic\b", "", t, flags=re.IGNORECASE)
    t = " ".join(t.replace(" ,", ",").split()).strip(" ,.")
    if not t:
        t = prompt.strip()
    t = t[0].upper() + t[1:]
    if _STYLE_WORDS.search(t) and not wants_real:
        return f"{t}. Detailed, coherent composition, clean lines, balanced colours."
    return f"{t}. {_PHOTO_CUES[0].upper()}{_PHOTO_CUES[1:]}."


def clean_enhanced(text: str, original: str) -> str | None:
    t = re.sub(r"<think>[\s\S]*?</think>", "", text or "").strip()
    t = re.sub(r"^(?:prompt|image prompt)\s*:\s*", "", t, flags=re.IGNORECASE).strip().strip('"').strip()
    t = " ".join(t.split())
    if len(t.split()) < 8 or len(t) > 1400:
        return None
    return t


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
    aspect: str = ""
    prompt_used: str = ""            # what the image model actually got (enhanced or as typed)
    prompt_source: str = ""          # chat model | rules | as typed
    profile: str = ""
    timings: dict = field(default_factory=dict)
    state: str = "queued"           # queued | enhancing | preparing | loading | rendering | decoding | done | failed | cancelled
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
            "prompt_used": self.prompt_used or self.prompt, "profile": self.profile,
            "prompt_source": self.prompt_source or "as typed",
            "timings": {k: round(v, 1) for k, v in self.timings.items() if not k.startswith("_")},
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
    job.percent = 20 + 70 * min(1.0, step / total)
    job.stage = f"Rendering · {step} of {total}"
    if m.lastindex and m.lastindex >= 4:
        rate = m.group(3)
        job.message = f"{rate} {m.group(4)}"


def build_command(setup: Setup, job: Job, output: Path, *, supported: set[str] | None = None,
                  low_vram: bool = False) -> list[str]:
    prof = profile_for(setup.model)
    supported = cli_flags(setup.cli) if supported is None else supported
    given = {a.split("=", 1)[0] for a in setup.extra_args if a.startswith("-")}
    cmd = [str(setup.cli), "--diffusion-model", str(setup.model), "--prompt", job.prompt_used or job.prompt,
           "--output", str(output), "--width", str(job.width), "--height", str(job.height),
           "--steps", str(job.steps)]
    if setup.text_encoder:
        cmd += ["--llm", str(setup.text_encoder)]
    if setup.vae:
        cmd += ["--vae", str(setup.vae)]
    if job.seed is not None:
        cmd += ["--seed", str(job.seed)]

    def add(flag: str, *values: str, need_support: bool = True) -> None:
        if flag in given:
            return  # the user's extra_args decide
        if need_support and supported and flag not in supported:
            return
        if need_support and not supported:
            return  # unknown binary: only pass flags every build has
        cmd.extend([flag, *values])

    add("--cfg-scale", f"{prof.cfg_scale:g}", need_support=False)
    add("--sampling-method", prof.sampler, need_support=False)
    if prof.sigmas:
        add("--sigmas", prof.sigmas)
    for flag in speed_flags(supported, low_vram=low_vram):
        add(*flag)
    cmd += setup.extra_args
    return cmd


def _run_cli(setup: Setup, job: Job, output: Path, *, low_vram: bool, timeout_s: float) -> None:
    t_load = time.time()
    t_render = None
    cmd = build_command(setup, job, output, low_vram=low_vram)
    job.log_tail = []
    assert setup.cli is not None
    proc = subprocess.Popen(
        cli_argv(setup.cli, *cmd[1:]), cwd=str(setup.cli.parent),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=_NO_WINDOW,
    )
    job._proc = proc
    deadline = time.time() + timeout_s
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            for piece in re.split(r"\r", raw.rstrip("\r\n")):
                piece = piece.strip()
                if not piece:
                    continue
                job.log_tail.append(piece[-300:])
                if len(job.log_tail) > 40:
                    job.log_tail.pop(0)
                low = piece.lower()
                if "sampling" in low or _STEP_RE.search(piece) or _PCT_RE.search(piece):
                    if job.state != "rendering":
                        t_render = time.time()
                        job.timings["load_s"] = t_render - t_load
                        job._set("rendering", "Rendering", 20)
                    _parse_progress(piece, job)
                elif "decod" in low and job.state == "rendering":
                    job.timings["render_s"] = time.time() - (t_render or t_load)
                    job._set("decoding", "Finishing", 92)
            if time.time() > deadline:
                proc.kill()
                raise TimeoutError(f"image generation exceeded {int(timeout_s)} s")
            if job._cancel:
                raise _Cancelled()
        code = proc.wait()
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        job._proc = None
    if job._cancel:
        raise _Cancelled()
    if code != 0 or not output.is_file():
        tail = "\n".join(job.log_tail[-8:]) or "image generator failed"
        raise RuntimeError(tail[-1200:])


def run_job(job: Job, *, timeout_s: float | None = None, use_server: bool | None = None) -> None:
    """Blocking render; call from a worker thread. The caller owns the GPU handoff.

    Uses the resident sd-server when it is available (weights stay loaded between
    pictures), else sd-cli. A render that runs out of GPU memory is retried once in
    low-VRAM mode, which is then remembered."""
    import image_server  # local: image_server imports this module

    timeout_s = timeout_s or float(os.environ.get("HIVI_IMAGE_TIMEOUT_S", "900"))
    with _RUN_LOCK:
        job.started_at = job.started_at or time.time()
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
        job.profile = profile_for(setup.model).name
        if use_server is None:
            use_server = (image_server.enabled() and image_server.server_binary(setup) is not None
                          and time.time() >= image_server.SERVER.broken_until)
        low_vram = learned_low_vram()
        output = images_dir() / f"{secrets.token_urlsafe(18).replace('-', '_')}.png"
        job._set("loading", "Loading the image model", max(job.percent, 12))
        try:
            for attempt in (1, 2):
                t0 = time.time()
                try:
                    if use_server:
                        if image_server.SERVER.alive():
                            job.timings["_server_warm"] = 1.0
                        try:
                            image_server.SERVER.render(setup, job, output, low_vram=low_vram, timeout_s=timeout_s)
                        except RuntimeError as e:
                            if "while starting" in str(e) or "did not start" in str(e) or "not found" in str(e):
                                # Server mode does not work with this build: sd-cli for 10 minutes.
                                image_server.SERVER.broken_until = time.time() + 600
                                use_server = False
                                job.message = "sd-server unavailable, using sd-cli"
                                _run_cli(setup, job, output, low_vram=low_vram, timeout_s=timeout_s)
                            else:
                                raise
                        if "render_s" not in job.timings and "_t_render" in job.timings:
                            job.timings["render_s"] = time.time() - job.timings["_t_render"]
                        if "load_s" not in job.timings:
                            job.timings["load_s"] = time.time() - t0
                    else:
                        _run_cli(setup, job, output, low_vram=low_vram, timeout_s=timeout_s)
                    break
                except RuntimeError as e:
                    if attempt == 1 and not low_vram and _is_oom(str(e)):
                        low_vram = True
                        remember_low_vram(setup.model)
                        image_server.SERVER.stop()
                        job._set("loading", "Out of GPU memory · retrying in low-VRAM mode", 12)
                        job.step = job.total_steps = 0
                        continue
                    raise
            if not output.is_file():
                raise RuntimeError("the image generator finished without writing an image")
            job.filename = output.name
            job.timings["total_s"] = time.time() - (job.started_at or time.time())
            _write_sidecar(output, job, setup)
            job._set("done", "Done", 100)
        except _Cancelled:
            job._set("cancelled", "Cancelled", 0)
        except (RuntimeError, TimeoutError, OSError) as e:
            job.error = str(e)
            job._set("failed", "Failed", 0)
        finally:
            for k in [k for k in job.timings if k.startswith("_")]:
                job.timings.pop(k, None)
            job._proc = None
            job.finished_at = time.time()


class _Cancelled(Exception):
    pass


def _write_sidecar(output: Path, job: Job, setup: Setup) -> None:
    try:
        meta = {"prompt": job.prompt, "prompt_used": job.prompt_used or job.prompt,
                "width": job.width, "height": job.height, "steps": job.steps, "seed": job.seed,
                "model": setup.model.name if setup.model else None, "profile": job.profile,
                "timings": {k: round(v, 1) for k, v in job.timings.items() if not k.startswith("_")},
                "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        output.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def new_job(prompt: str, *, width: int, height: int, steps: int | None = None, seed: int | None = None,
            aspect: str = "") -> Job:
    if steps is None:
        steps = profile_for(discover().model).steps
    if seed is None:
        seed = random.randint(1, 2**31 - 1)   # sd-cli's default is a fixed 42: every Regenerate looked the same
    job = Job(id=uuid.uuid4().hex[:16], prompt=prompt, width=width, height=height, steps=steps, seed=seed,
              aspect=aspect)
    _remember(job)
    return job


def list_images(limit: int = 60) -> list[dict[str, Any]]:
    out = []
    for p in sorted(images_dir().glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        meta = {}
        side = p.with_suffix(".json")
        if side.is_file():
            try:
                meta = yaml.safe_load(side.read_text(encoding="utf-8")) or {}   # JSON is valid YAML
            except (OSError, yaml.YAMLError):
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        out.append({"image_url": f"/api/images/{p.name}", "filename": p.name,
                    "prompt": meta.get("prompt", ""), "seed": meta.get("seed"),
                    "created": meta.get("created"), "size_kb": round(p.stat().st_size / 1024)})
    return out
