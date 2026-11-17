"""IMG-1: local image generation: discovery, model settings, GPU handoff, progress."""
from __future__ import annotations

import asyncio
import json
import stat
import textwrap
import threading
import time

import httpx
import image_gen
import image_server
import pytest
import server

_FAKE_SERVER = textwrap.dedent('''\
    #!/usr/bin/env python3
    import base64, json, os, sys, threading, time, uuid
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    a = sys.argv[1:]
    if a == ["--help"]:
        print("  --listen-port\\n  --flash-attn\\n  --diffusion-fa\\n  --auto-fit\\n  --mmap\\n  --offload-to-cpu\\n"
              "  --vae-tiling\\n  --sigmas\\n  --cfg-scale\\n  --sampling-method\\n  --steps")
        sys.exit(0)
    if os.environ.get("FAKE_SERVER_DIE"):
        print("fatal: cannot load model", flush=True); sys.exit(3)
    log = os.environ.get("FAKE_SD_ARGS")
    if log:
        open(log, "a").write("SERVER " + " ".join(a) + "\\n")
    port = int(a[a.index("--listen-port") + 1])
    jobs = {}
    loaded = {"done": False}
    def work(jid, body):
        jobs[jid]["status"] = "generating"
        if not loaded["done"]:
            print("[INFO ] loading diffusion model", flush=True); time.sleep(0.3); loaded["done"] = True
        steps = body["sample_params"]["sample_steps"]
        print("[INFO ] sampling using Euler method", flush=True)
        for i in range(1, steps + 1):
            if jobs[jid]["status"] == "cancelled":
                return
            time.sleep(float(os.environ.get("FAKE_SD_SLEEP", "0.02")))
            print(f"  |{'=' * i}| {i}/{steps} - 0.05s/it", flush=True)
        print("[INFO ] decoding latent", flush=True)
        png = b"\\x89PNG\\r\\n\\x1a\\n" + body["prompt"].encode()[:40]
        jobs[jid].update(status="completed", result={"output_format": "png",
                         "images": [{"index": 0, "b64_json": base64.b64encode(png).decode()}]})
    class H(BaseHTTPRequestHandler):
        def log_message(self, *x): pass
        def send(self, code, obj):
            b = json.dumps(obj).encode(); self.send_response(code)
            self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            if self.path == "/sdcpp/v1/capabilities": return self.send(200, {"model": {"name": "x"}})
            jid = self.path.rsplit("/", 1)[-1]
            return self.send(200, {"id": jid, **jobs[jid]}) if jid in jobs else self.send(404, {})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/sdcpp/v1/img_gen":
                jid = "job_" + uuid.uuid4().hex[:8]; jobs[jid] = {"status": "queued", "result": None, "error": None}
                threading.Thread(target=work, args=(jid, body), daemon=True).start()
                return self.send(202, {"id": jid, "status": "queued"})
            if self.path.endswith("/cancel"):
                jid = self.path.split("/")[-2]; jobs[jid]["status"] = "cancelled"; return self.send(200, {})
            self.send(404, {})
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
''')

_FAKE_SD = textwrap.dedent('''\
    #!/usr/bin/env python3
    import os, sys, time
    a = sys.argv[1:]
    known = {"--diffusion-model", "--prompt", "--output", "--width", "--height", "--steps", "--llm",
             "--vae", "--seed", "--cfg-scale", "--sampling-method", "--sigmas", "--diffusion-fa",
             "--offload-to-cpu", "--vae-tiling", "--backend", "--flash-attn", "--auto-fit", "--mmap"}
    if a == ["--help"]:
        print("usage: sd-cli [options]\\n" + "\\n".join("  " + k + "   help text" for k in sorted(known)))
        sys.exit(0)
    for x in a:
        if x.startswith("--") and x not in known:
            print("error: unknown argument: " + x); sys.exit(1)
    log = os.environ.get("FAKE_SD_ARGS")
    if log:
        open(log, "w").write(" ".join(a))
    out = a[a.index("--output") + 1]; steps = int(a[a.index("--steps") + 1])
    if os.environ.get("FAKE_SD_FAIL") and not (os.environ.get("FAKE_SD_FAIL") == "oom-once"
                                               and "--offload-to-cpu" in a):
        print("ggml_cuda: CUDA error: out of memory"); sys.exit(1)
    print("[INFO ] loading diffusion model", flush=True)
    print("[INFO ] sampling using Euler method", flush=True)
    for i in range(1, steps + 1):
        time.sleep(float(os.environ.get("FAKE_SD_SLEEP", "0.02")))
        print(f"  |{'=' * i}| {i}/{steps} - 0.05s/it", flush=True)
    print("[INFO ] decoding latent", flush=True)
    open(out, "wb").write(b"\\x89PNG\\r\\n\\x1a\\n" + b"0" * 64)
''')


@pytest.fixture
def image_setup(tmp_path, monkeypatch, request):
    root = tmp_path / "Models" / "Image"
    (root / "text_encoders").mkdir(parents=True)
    cli = root / "sd-cli"
    cli.write_text(_FAKE_SD, encoding="utf-8")
    cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    (root / "qwen-image-2.1-turbo-Q4_K_M.gguf").write_bytes(b"0")
    (root / "qwen-image-2.1-turbo-Q8_0.gguf").write_bytes(b"0")
    (root / "text_encoders" / "Qwen3VL-8B-Instruct-Q4_K_M.gguf").write_bytes(b"0")
    (root / "qwen_image_2.1_vae_bf16.safetensors").write_bytes(b"0")
    monkeypatch.setenv("HIVI_MODELS_ROOT", str(tmp_path / "Models"))
    monkeypatch.setenv("HIVI_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(image_gen, "config_file", lambda: tmp_path / "image.yaml")
    for key in ("HIVI_SD_CLI", "HIVI_IMAGE_MODEL", "HIVI_IMAGE_LLM", "HIVI_IMAGE_VAE", "HIVI_IMAGE_ARGS"):
        monkeypatch.delenv(key, raising=False)
    if request.node.get_closest_marker("server"):
        srv = root / "sd-server"
        srv.write_text(_FAKE_SERVER, encoding="utf-8")
        srv.chmod(srv.stat().st_mode | stat.S_IEXEC)
    image_gen._JOBS.clear()
    image_gen._HELP_CACHE.clear()
    image_server.SERVER.stop()
    image_server.SERVER.broken_until = 0.0
    server.IMAGE_GPU.__init__()
    saved = (server.STATE.model_id, server.STATE.status)
    yield root
    server.IMAGE_GPU.cancel_timer()
    server.IMAGE_GPU.__init__()
    server.STATE.model_id, server.STATE.status = saved
    image_server.SERVER.stop()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://hivi")


def test_discovery_without_environment_variables(image_setup):
    st = image_gen.status()
    assert st["ready"] is True and st["missing"] == []
    assert st["model"] == "qwen-image-2.1-turbo-Q4_K_M.gguf"
    assert st["text_encoder"] == "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    assert st["profile"] == "Qwen-Image 2.1 Turbo" and st["steps"] == 8 and st["cfg_scale"] == 1.0
    assert (st["default_width"], st["default_height"]) == (1024, 1024)


def test_text_encoder_found_by_folder_whatever_its_name(image_setup):
    (image_setup / "text_encoders" / "Qwen3VL-8B-Instruct-Q4_K_M.gguf").unlink()
    (image_setup / "text_encoders" / "qwen_vl_fp8_scaled.safetensors").write_bytes(b"0")
    assert image_gen.status()["text_encoder"] == "qwen_vl_fp8_scaled.safetensors"


@pytest.mark.parametrize(("name", "steps", "cfg", "sigmas"), [
    ("qwen-image-2.1-turbo-Q4_K_M.gguf", 8, 1.0, "qwen-image-2.1-turbo"),
    ("qwen_image_2.1-Q4_K.gguf", 20, 6.0, None),
    ("qwen-image-2512-Q4_K_M.gguf", 20, 2.5, None),
    ("z_image_turbo-Q8_0.gguf", 8, 1.0, None),
])
def test_profiles(name, steps, cfg, sigmas, tmp_path, monkeypatch):
    monkeypatch.setattr(image_gen, "config_file", lambda: tmp_path / "none.yaml")
    p = image_gen.profile_for(tmp_path / name)
    assert (p.steps, p.cfg_scale, p.sigmas) == (steps, cfg, sigmas)


def test_command_uses_profile_and_supported_speed_flags(image_setup, tmp_path):
    s = image_gen.discover()
    job = image_gen.new_job("x", width=1024, height=1024)
    assert job.steps == 8 and job.seed is not None          # random seed, not sd-cli's fixed 42
    cmd = image_gen.build_command(s, job, tmp_path / "o.png")
    joined = " ".join(cmd)
    for flag in ("--cfg-scale 1", "--sampling-method euler", "--sigmas qwen-image-2.1-turbo",
                 "--mmap", "--steps 8"):
        assert flag in joined, flag
    # --auto-fit and --flash-attn auto exist: no forced offload, no deprecated flag
    for flag in ("--offload-to-cpu", "--vae-tiling", "--diffusion-fa"):
        assert flag not in cmd, flag
    low = image_gen.build_command(s, job, tmp_path / "o.png", low_vram=True)
    assert "--offload-to-cpu" in low and "--vae-tiling" in low
    # An older build without auto-fit / flash-attn gets the explicit flags
    older = {"--cfg-scale", "--sampling-method", "--diffusion-fa", "--offload-to-cpu", "--vae-tiling"}
    old = image_gen.build_command(s, job, tmp_path / "o.png", supported=older)
    assert "--diffusion-fa" in old and "--offload-to-cpu" in old and "--sigmas" not in old


def test_extra_args_win_over_defaults(image_setup, tmp_path):
    (tmp_path / "image.yaml").write_text(
        "extra_args: '--cfg-scale 2.0 --sampling-method euler --offload-to-cpu'\n", encoding="utf-8")
    s = image_gen.discover()
    cmd = image_gen.build_command(s, image_gen.new_job("x", width=1024, height=1024), tmp_path / "o.png")
    assert cmd.count("--cfg-scale") == 1 and cmd[cmd.index("--cfg-scale") + 1] == "2.0"
    assert cmd.count("--offload-to-cpu") == 1


def test_sizes():
    assert image_gen.valid_size(1024, 1024) and image_gen.valid_size(1344, 768)
    assert not image_gen.valid_size(1000, 1000) and not image_gen.valid_size(1536, 1536)


def test_job_reports_progress_and_timings(image_setup, tmp_path, monkeypatch):
    args = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_SD_ARGS", str(args))
    job = image_gen.new_job("a bunny", width=1152, height=864)
    job.prompt_used = "a detailed bunny prompt"
    image_gen.run_job(job)
    assert job.state == "done" and job.percent == 100 and job.step == 8 == job.total_steps
    assert "--prompt a detailed bunny prompt" in args.read_text()
    assert {"load_s", "render_s", "total_s"} <= set(job.timings)
    side = json.loads((image_gen.images_dir() / job.filename).with_suffix(".json").read_text())
    assert side["prompt_used"] == "a detailed bunny prompt" and side["profile"] == "Qwen-Image 2.1 Turbo"
    assert image_gen.list_images()[0]["prompt"] == "a bunny"


def test_failed_render_keeps_the_log_tail(image_setup, monkeypatch):
    monkeypatch.setenv("FAKE_SD_FAIL", "1")
    job = image_gen.new_job("x", width=1024, height=1024)
    image_gen.run_job(job)
    assert job.state == "failed" and "out of memory" in job.error


def test_cancel_kills_the_render(image_setup, monkeypatch):
    monkeypatch.setenv("FAKE_SD_SLEEP", "0.4")
    job = image_gen.new_job("x", width=1024, height=1024, steps=20)
    t = threading.Thread(target=image_gen.run_job, args=(job,))
    t.start()
    for _ in range(200):
        if job.state == "rendering":
            break
        time.sleep(0.02)
    job.cancel()
    t.join(timeout=10)
    assert job.state == "cancelled" and job.filename is None


def test_enhanced_prompt_cleanup():
    assert image_gen.needs_enhancing("a bunny in the rain")
    assert not image_gen.needs_enhancing("render exactly this: a bunny")
    out = image_gen.clean_enhanced('<think>x</think>Prompt: "A small white rabbit sits in neon rain on a wet street"', "a")
    assert out == "A small white rabbit sits in neon rain on a wet street"
    assert image_gen.clean_enhanced("ok", "a") is None


def _fake_runtime(monkeypatch, loaded="qwen36"):
    """Stand-ins for the engine so the GPU handoff can be checked without one."""
    calls = []
    server.STATE.model_id = loaded
    server.STATE.status = "ready" if loaded else "idle"

    def stop():
        calls.append("stop")
        server.STATE.model_id = None
        server.STATE.status = "idle"

    class _State:
        status = "ready"
        detail = "ok"

    def select(model):
        calls.append(f"load:{model.id}")
        server.STATE.model_id = model.id
        server.STATE.status = "ready"
        return _State()

    class _M:
        def __init__(self, mid):
            self.id = mid
            self.name = mid.upper()

    monkeypatch.setattr(server, "stop_runtime", stop)
    monkeypatch.setattr(server, "select_model", select)
    monkeypatch.setattr(server, "get_model", lambda mid: _M(mid) if mid else None)
    monkeypatch.setattr(server, "_is_loaded", lambda m: server.STATE.model_id == m.id)
    return calls


async def _wait_job(client, jid):
    for _ in range(400):
        d = (await client.get(f"/api/images/jobs/{jid}")).json()
        if d["done"]:
            return d
        await asyncio.sleep(0.02)
    raise AssertionError("job did not finish")


async def test_gpu_handoff_parks_then_reloads_after_delay(image_setup, tmp_path, monkeypatch):
    (tmp_path / "image.yaml").write_text("reload_delay_s: 0.3\nenhance_prompt: false\n", encoding="utf-8")
    calls = _fake_runtime(monkeypatch)
    async with _client() as client:
        jid = (await client.post("/api/images/generate", json={"prompt": "a fox", "aspect": "3:4"})).json()["id"]
        d = await _wait_job(client, jid)
        assert d["state"] == "done" and (d["width"], d["height"]) == (864, 1152) and d["steps"] == 8
        assert calls == ["stop"]                         # the image is shown before any reload
        st = (await client.get("/api/status")).json()
        assert st["status"] == "paused" and st["model_id"] == "qwen36"   # Send keeps working
        await asyncio.sleep(0.6)
        assert calls == ["stop", "load:qwen36"]
        assert server.IMAGE_GPU.parked is None
        img = await client.get(d["image_url"])
        assert img.status_code == 200


async def test_regenerate_within_delay_skips_reload(image_setup, tmp_path, monkeypatch):
    (tmp_path / "image.yaml").write_text("reload_delay_s: 5\nenhance_prompt: false\n", encoding="utf-8")
    calls = _fake_runtime(monkeypatch)
    async with _client() as client:
        for _ in range(2):
            jid = (await client.post("/api/images/generate", json={"prompt": "a fox"})).json()["id"]
            await _wait_job(client, jid)
        assert calls == ["stop"]                         # one unload for two images
        server.IMAGE_GPU.cancel_timer()


async def test_chat_request_wakes_the_parked_model(image_setup, tmp_path, monkeypatch):
    (tmp_path / "image.yaml").write_text("reload_delay_s: 30\nenhance_prompt: false\n", encoding="utf-8")
    calls = _fake_runtime(monkeypatch)
    async with _client() as client:
        jid = (await client.post("/api/images/generate", json={"prompt": "a fox"})).json()["id"]
        await _wait_job(client, jid)
        assert calls == ["stop"]
        await server.IMAGE_GPU.reload_now()             # what the proxy does on a chat message
        assert calls == ["stop", "load:qwen36"] and server.IMAGE_GPU.reload_at is None


async def test_nothing_can_load_onto_the_gpu_during_a_render(image_setup, tmp_path, monkeypatch):
    (tmp_path / "image.yaml").write_text("reload_delay_s: 30\nenhance_prompt: false\n", encoding="utf-8")
    monkeypatch.setenv("FAKE_SD_SLEEP", "0.1")
    _fake_runtime(monkeypatch)
    async with _client() as client:
        jid = (await client.post("/api/images/generate", json={"prompt": "a fox"})).json()["id"]
        for _ in range(200):
            if server.IMAGE_GPU.busy:
                break
            await asyncio.sleep(0.01)
        assert server.SWITCH_LOCK.locked()
        r = await client.post("/api/select", json={"id": "qwen36"})
        assert r.status_code == 409 and "drawing" in r.json()["detail"]
        assert (await client.get("/api/status")).json()["status"] == "drawing"
        await _wait_job(client, jid)
        assert not server.SWITCH_LOCK.locked()
        server.IMAGE_GPU.cancel_timer()


async def test_api_validation(image_setup):
    async with _client() as client:
        assert (await client.post("/api/images/generate", json={"prompt": "x", "width": 1000})).status_code == 400
        assert (await client.get("/api/images/jobs/nope")).status_code == 404
        (image_setup / "sd-cli").unlink()
        r = await client.post("/api/images/generate", json={"prompt": "x"})
        assert r.status_code == 503 and "sd-cli" in r.json()["detail"]


def test_polish_prompt_without_a_chat_model():
    out = image_gen.polish_prompt("get me a image of mountains and rivers flowing from it a realistic 4k")
    assert out.startswith("Mountains and rivers") and "4k" not in out and "photograph" in out.lower()
    anime = image_gen.polish_prompt("make me an anime girl on a rooftop at sunset")
    assert anime.startswith("An anime girl") and "photograph" not in anime.lower()


async def test_prompt_falls_back_to_rules_when_chat_model_is_parked(image_setup, tmp_path, monkeypatch):
    (tmp_path / "image.yaml").write_text("reload_delay_s: 30\n", encoding="utf-8")
    _fake_runtime(monkeypatch)
    async with _client() as client:
        jid = (await client.post("/api/images/generate", json={"prompt": "a cozy cafe"})).json()["id"]
        first = await _wait_job(client, jid)      # chat model "ready" but backend is fake → rules
        assert first["prompt_source"] == "rules" and "photograph" in first["prompt_used"].lower()
        jid = (await client.post("/api/images/generate", json={"prompt": "a cozy cafe"})).json()["id"]
        second = await _wait_job(client, jid)     # parked now → rules, no backend call
        assert second["prompt_source"] == "rules"


def test_oom_retries_once_in_low_vram_mode_and_remembers(image_setup, monkeypatch):
    monkeypatch.setenv("FAKE_SD_FAIL", "oom-once")
    job = image_gen.new_job("x", width=1024, height=1024)
    image_gen.run_job(job, use_server=False)
    assert job.state == "done"
    assert image_gen.learned_low_vram() is True
    cmd = image_gen.build_command(image_gen.discover(), job, image_gen.images_dir() / "o.png",
                                  low_vram=image_gen.learned_low_vram())
    assert "--offload-to-cpu" in cmd


@pytest.mark.server
def test_server_mode_keeps_weights_loaded_between_pictures(image_setup, tmp_path, monkeypatch):
    args = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_SD_ARGS", str(args))
    first = image_gen.new_job("a fox", width=1024, height=1024)
    image_gen.run_job(first)
    assert first.state == "done", first.error
    assert image_server.SERVER.alive()
    second = image_gen.new_job("a cat", width=864, height=1152)
    image_gen.run_job(second)
    assert second.state == "done" and second.step == 8
    log = args.read_text()
    assert log.count("SERVER ") == 1                    # started once, reused
    assert "--sigmas qwen-image-2.1-turbo" in log and "--offload-to-cpu" not in log
    data = (image_gen.images_dir() / second.filename).read_bytes()
    assert data.startswith(b"\x89PNG") and b"a cat" in data
    image_server.SERVER.stop()
    assert not image_server.SERVER.alive()


@pytest.mark.server
def test_broken_server_falls_back_to_cli(image_setup, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_DIE", "1")
    job = image_gen.new_job("x", width=1024, height=1024)
    image_gen.run_job(job)
    assert job.state == "done"
    assert image_server.SERVER.broken_until > time.time()


@pytest.mark.server
def test_server_cancel(image_setup, monkeypatch):
    monkeypatch.setenv("FAKE_SD_SLEEP", "0.3")
    job = image_gen.new_job("x", width=1024, height=1024, steps=20)
    t = threading.Thread(target=image_gen.run_job, args=(job,))
    t.start()
    for _ in range(300):
        if job.state == "rendering":
            break
        time.sleep(0.02)
    job.cancel()
    t.join(timeout=15)
    assert job.state == "cancelled"


@pytest.mark.server
async def test_server_stops_before_the_chat_model_returns(image_setup, tmp_path, monkeypatch):
    (tmp_path / "image.yaml").write_text("reload_delay_s: 30\nenhance_prompt: false\n", encoding="utf-8")
    calls = _fake_runtime(monkeypatch)
    async with _client() as client:
        jid = (await client.post("/api/images/generate", json={"prompt": "a fox"})).json()["id"]
        d = await _wait_job(client, jid)
        assert d["state"] == "done", d["error"]
        assert image_server.SERVER.alive()
        st = (await client.get("/api/images/status")).json()
        assert st["server"]["running"] is True
        await server.IMAGE_GPU.reload_now()
        assert not image_server.SERVER.alive() and calls == ["stop", "load:qwen36"]
