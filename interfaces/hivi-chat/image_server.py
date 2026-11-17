"""Keep stable-diffusion.cpp's ``sd-server`` running during an image session.

``sd-cli`` loads ~9 GB of weights from disk for every picture (about 40 s on an
8 GB laptop). ``sd-server`` loads them once; later pictures only pay for the
render. It lives only while the chat model is parked (``server._ImageGpu``): the
chat model and the image weights do not both fit in 32 GB of RAM, so the server
is stopped before the chat model comes back.

API used (stable-diffusion.cpp ``examples/server/api.md``): ``POST
/sdcpp/v1/img_gen`` → job id, ``GET /sdcpp/v1/jobs/{id}`` → status and base64 PNG,
``POST /sdcpp/v1/jobs/{id}/cancel``. Progress is not in the API, so the server's
own log lines are parsed for sampling steps, as with ``sd-cli``.
"""
from __future__ import annotations

import base64
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import image_gen

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_START_TIMEOUT_S = 120.0


def server_binary(setup: image_gen.Setup) -> Path | None:
    cfg = image_gen.load_config()
    custom = cfg.get("server")
    if custom and Path(custom).is_file():
        return Path(custom)
    if setup.cli is None:
        return None
    for name in ("sd-server.exe", "sd-server"):
        cand = setup.cli.parent / name
        if cand.is_file():
            return cand
    return None


def enabled() -> bool:
    v = image_gen.load_config().get("server_mode", "auto")
    return str(v).lower() not in ("false", "off", "no", "0")


def _free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return s.getsockname()[1]
    return preferred


class ImageServer:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self.key: tuple | None = None        # model files + flags it was started with
        self.log_tail: list[str] = []
        self.current: image_gen.Job | None = None
        self.started_at: float | None = None
        self.broken_until = 0.0               # after a failed start, use sd-cli for a while

    # -- lifecycle ---------------------------------------------------------- #

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        with self._lock:
            proc, self.proc = self.proc, None
            self.port = self.key = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=8)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                except OSError:
                    pass

    def command(self, setup: image_gen.Setup, binary: Path, port: int, *, low_vram: bool) -> list[str]:
        prof = image_gen.profile_for(setup.model)
        supported = image_gen.cli_flags(binary)
        cmd = [str(binary), "--listen-ip", "127.0.0.1", "--listen-port", str(port),
               "--diffusion-model", str(setup.model)]
        if setup.text_encoder:
            cmd += ["--llm", str(setup.text_encoder)]
        if setup.vae:
            cmd += ["--vae", str(setup.vae)]
        # Defaults for every request: the request below only sets prompt, size, seed, steps, cfg.
        cmd += ["--cfg-scale", f"{prof.cfg_scale:g}", "--sampling-method", prof.sampler,
                "--steps", str(prof.steps)]
        if prof.sigmas and "--sigmas" in supported:
            cmd += ["--sigmas", prof.sigmas]
        given = {a.split("=", 1)[0] for a in setup.extra_args if a.startswith("-")}
        for flag in image_gen.speed_flags(supported, low_vram=low_vram):
            if flag[0] not in given:
                cmd += flag
        cmd += list(setup.extra_args)
        return cmd

    def ensure(self, setup: image_gen.Setup, job: image_gen.Job, *, low_vram: bool = False) -> str:
        """Start (or reuse) the server; returns its base URL. Raises RuntimeError."""
        binary = server_binary(setup)
        if binary is None:
            raise RuntimeError("sd-server not found next to sd-cli")
        key = (str(setup.model), str(setup.text_encoder), str(setup.vae), tuple(setup.extra_args), low_vram)
        with self._lock:
            if self.alive() and self.key == key and self.port:
                return f"http://127.0.0.1:{self.port}"
            self.stop()
            port = _free_port(int(image_gen.load_config().get("server_port", 7873)))
            cmd = self.command(setup, binary, port, low_vram=low_vram)
            self.log_tail = []
            self.proc = subprocess.Popen(
                image_gen.cli_argv(binary, *cmd[1:]), cwd=str(binary.parent),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=_NO_WINDOW,
            )
            self.port, self.key, self.started_at = port, key, time.time()
            threading.Thread(target=self._read_log, args=(self.proc,), daemon=True,
                             name="hivi-sd-server-log").start()
        base = f"http://127.0.0.1:{port}"
        t_start = time.time()
        deadline = t_start + _START_TIMEOUT_S
        while time.time() < deadline:
            if not self.alive():
                tail = "\n".join(self.log_tail[-8:])
                self.stop()
                raise RuntimeError(f"sd-server exited while starting:\n{tail}")
            if job._cancel:
                raise image_gen._Cancelled()
            try:
                r = httpx.get(f"{base}/sdcpp/v1/capabilities", timeout=2.0)
                if r.status_code < 500:
                    job.timings["start_s"] = time.time() - t_start
                    return base
            except httpx.HTTPError:
                pass
            time.sleep(0.4)
        self.stop()
        raise RuntimeError("sd-server did not start in time")

    def _read_log(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            for piece in re.split(r"\r", raw.rstrip("\r\n")):
                piece = piece.strip()
                if not piece:
                    continue
                self.log_tail.append(piece[-300:])
                if len(self.log_tail) > 60:
                    self.log_tail.pop(0)
                job = self.current
                if job is None:
                    continue
                low = piece.lower()
                if "sampling" in low or image_gen._STEP_RE.search(piece) or image_gen._PCT_RE.search(piece):
                    self._mark_render(job)
                    image_gen._parse_progress(piece, job)
                elif "decod" in low and job.state == "rendering":
                    job.timings["render_s"] = time.time() - job.timings.get("_t_render", time.time())
                    job._set("decoding", "Finishing", 92)

    @staticmethod
    def _mark_render(job: image_gen.Job) -> None:
        now = time.time()
        if "_t_render" not in job.timings:
            job.timings["_t_render"] = now
            job.timings["load_s"] = now - job.timings.get("_t_submit", now)
        if job.state not in ("rendering", "decoding"):
            job._set("rendering", "Rendering", 20)

    # -- one picture -------------------------------------------------------- #

    def render(self, setup: image_gen.Setup, job: image_gen.Job, output: Path, *, low_vram: bool,
               timeout_s: float) -> None:
        base = self.ensure(setup, job, low_vram=low_vram)
        prof = image_gen.profile_for(setup.model)
        body: dict[str, Any] = {
            "prompt": job.prompt_used or job.prompt,
            "negative_prompt": "",
            "width": job.width,
            "height": job.height,
            "seed": job.seed if job.seed is not None else -1,
            "batch_count": 1,
            "sample_params": {
                "sample_method": prof.sampler,
                "sample_steps": job.steps,
                "guidance": {"txt_cfg": prof.cfg_scale},
            },
            "output_format": "png",
        }
        self.current = job
        job.timings["_t_submit"] = time.time()
        warm = bool(job.timings.get("_server_warm"))
        job._set("loading", "Starting" if warm else "Loading the image model", 14)
        try:
            r = httpx.post(f"{base}/sdcpp/v1/img_gen", json=body, timeout=30.0)
            if r.status_code >= 400:
                raise RuntimeError(f"sd-server refused the request: HTTP {r.status_code} {r.text[:300]}")
            sid = r.json().get("id")
            if not sid:
                raise RuntimeError("sd-server returned no job id")
            deadline = time.time() + timeout_s
            while True:
                if job._cancel:
                    try:
                        httpx.post(f"{base}/sdcpp/v1/jobs/{sid}/cancel", timeout=5.0)
                    except httpx.HTTPError:
                        pass
                    raise image_gen._Cancelled()
                if not self.alive():
                    raise RuntimeError("sd-server stopped:\n" + "\n".join(self.log_tail[-8:]))
                if time.time() > deadline:
                    raise TimeoutError(f"image generation exceeded {int(timeout_s)} s")
                s = httpx.get(f"{base}/sdcpp/v1/jobs/{sid}", timeout=10.0).json()
                status = s.get("status")
                if status == "generating" and warm and job.state == "loading":
                    self._mark_render(job)   # weights already loaded: generating = rendering
                if status == "completed":
                    images = ((s.get("result") or {}).get("images") or [])
                    if not images or not images[0].get("b64_json"):
                        raise RuntimeError("sd-server finished without an image")
                    output.write_bytes(base64.b64decode(images[0]["b64_json"]))
                    return
                if status in ("failed", "cancelled"):
                    err = s.get("error")
                    if isinstance(err, dict):
                        err = err.get("message") or str(err)
                    tail = "\n".join(self.log_tail[-6:])
                    raise RuntimeError(f"{err or status}\n{tail}".strip())
                time.sleep(0.4)
        finally:
            self.current = None

    def status(self) -> dict[str, Any]:
        return {"running": self.alive(), "port": self.port,
                "uptime_s": round(time.time() - self.started_at) if self.alive() and self.started_at else None}


SERVER = ImageServer()
