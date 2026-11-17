"""IMG-1: local image generation jobs, discovery, GPU handoff and progress."""
from __future__ import annotations

import asyncio
import stat
import sys
import textwrap

import httpx
import image_gen
import pytest
import server

_FAKE_SD = textwrap.dedent('''\
    #!/usr/bin/env python3
    import sys, time, os
    a = sys.argv[1:]
    out = a[a.index("--output") + 1]; steps = int(a[a.index("--steps") + 1])
    if os.environ.get("FAKE_SD_FAIL"):
        print("ggml_cuda: CUDA error: out of memory"); sys.exit(1)
    print("[INFO ] loading diffusion model", flush=True)
    print("[INFO ] sampling using Euler method", flush=True)
    for i in range(1, steps + 1):
        time.sleep(float(os.environ.get("FAKE_SD_SLEEP", "0.05")))
        print(f"  |{'=' * i}| {i}/{steps} - 0.05s/it", flush=True)
    print("[INFO ] decoding latent", flush=True)
    open(out, "wb").write(b"\\x89PNG\\r\\n\\x1a\\n" + b"0" * 64)
''')


@pytest.fixture
def image_setup(tmp_path, monkeypatch):
    root = tmp_path / "Models" / "Image"
    (root / "text_encoders").mkdir(parents=True)
    cli = root / "sd-cli"
    cli.write_text(_FAKE_SD, encoding="utf-8")
    cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    (root / "qwen-image-2.1-Q4_K_M.gguf").write_bytes(b"0")
    (root / "qwen-image-2.1-Q8_0.gguf").write_bytes(b"0")
    (root / "text_encoders" / "Qwen3VL-Q4_K_M.gguf").write_bytes(b"0")
    (root / "qwen_image_vae.safetensors").write_bytes(b"0")
    monkeypatch.setenv("HIVI_MODELS_ROOT", str(tmp_path / "Models"))
    monkeypatch.setenv("HIVI_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(image_gen, "config_file", lambda: tmp_path / "image.yaml")
    # The fake CLI is a Python script, not a native Windows executable. Launch
    # only this fixture script through the active interpreter; leave the
    # production sd-cli.exe command untouched and preserve every CLI argument.
    real_build_command = image_gen.build_command

    def build_test_command(setup, job, output):
        command = real_build_command(setup, job, output)
        if setup.cli == cli:
            return [sys.executable, *command]
        return command

    monkeypatch.setattr(image_gen, "build_command", build_test_command)
    for key in ("HIVI_SD_CLI", "HIVI_IMAGE_MODEL", "HIVI_IMAGE_LLM", "HIVI_IMAGE_VAE", "HIVI_IMAGE_ARGS"):
        monkeypatch.delenv(key, raising=False)
    image_gen._JOBS.clear()
    return root


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://hivi")


def test_discovery_without_environment_variables(image_setup):
    st = image_gen.status()
    assert st["ready"] is True and st["missing"] == []
    assert st["model"] == "qwen-image-2.1-Q4_K_M.gguf"       # Q4_K_M preferred over Q8_0
    assert st["text_encoder"] == "Qwen3VL-Q4_K_M.gguf"
    assert st["vae"] == "qwen_image_vae.safetensors"
    assert st["cli"].endswith("sd-cli")


def test_discovery_reports_what_is_missing(image_setup):
    (image_setup / "text_encoders" / "Qwen3VL-Q4_K_M.gguf").unlink()
    st = image_gen.status()
    assert st["ready"] is False and any("text encoder" in m for m in st["missing"])


def test_config_file_overrides_and_extra_args(image_setup, tmp_path):
    (tmp_path / "image.yaml").write_text(
        f"model: {image_setup / 'qwen-image-2.1-Q8_0.gguf'}\nextra_args: '--offload-to-cpu --vae-tiling'\n"
        "unload_chat_model: false\n", encoding="utf-8")
    s = image_gen.discover()
    assert s.model.name == "qwen-image-2.1-Q8_0.gguf" and s.unload_chat is False
    job = image_gen.new_job("x", width=512, height=512, steps=4, seed=7)
    cmd = image_gen.build_command(s, job, tmp_path / "o.png")
    assert cmd[-2:] == ["--offload-to-cpu", "--vae-tiling"]
    assert "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "7"
    assert cmd[cmd.index("--llm") + 1].endswith("Qwen3VL-Q4_K_M.gguf")


def test_job_reports_progress_and_hands_the_gpu_over(image_setup):
    calls = []
    job = image_gen.new_job("a bunny", width=768, height=512, steps=3, seed=None)
    image_gen.run_job(job, unload_chat=lambda: (calls.append("unload"), "qwen36")[1],
                      reload_chat=lambda mid: calls.append(f"reload:{mid}"))
    assert job.state == "done" and job.percent == 100 and job.filename.endswith(".png")
    assert job.total_steps == 3 and job.step == 3
    assert calls == ["unload", "reload:qwen36"]
    assert (image_gen.images_dir() / job.filename).with_suffix(".json").is_file()
    d = job.to_dict()
    assert d["image_url"] == f"/api/images/{job.filename}" and d["done"] is True


def test_failed_render_keeps_the_log_tail_and_reloads_chat(image_setup, monkeypatch):
    monkeypatch.setenv("FAKE_SD_FAIL", "1")
    calls = []
    job = image_gen.new_job("x", width=512, height=512, steps=2, seed=None)
    image_gen.run_job(job, unload_chat=lambda: "m", reload_chat=lambda mid: calls.append(mid))
    assert job.state == "failed" and "out of memory" in job.error
    assert calls == ["m"]


def test_cancel_kills_the_render(image_setup, monkeypatch):
    monkeypatch.setenv("FAKE_SD_SLEEP", "0.4")
    job = image_gen.new_job("x", width=512, height=512, steps=20, seed=None)
    import threading
    t = threading.Thread(target=image_gen.run_job, args=(job,), kwargs={})
    t.start()
    for _ in range(100):
        if job.state == "rendering":
            break
        asyncio.run(asyncio.sleep(0.05))
    job.cancel()
    t.join(timeout=10)
    assert job.state == "cancelled" and job.filename is None


async def test_api_job_flow(image_setup, monkeypatch):
    monkeypatch.setattr(server, "_unload_chat_for_image", lambda: None)
    monkeypatch.setattr(server, "_reload_chat_after_image", lambda mid: None)
    async with _client() as client:
        assert (await client.get("/api/images/status")).json()["ready"] is True
        r = await client.post("/api/images/generate", json={"prompt": "a fox", "width": 640, "height": 640})
        assert r.status_code == 200
        jid = r.json()["id"]
        assert (await client.post("/api/images/generate", json={"prompt": "x", "width": 700})).status_code == 400
        for _ in range(100):
            d = (await client.get(f"/api/images/jobs/{jid}")).json()
            if d["done"]:
                break
            await asyncio.sleep(0.05)
        assert d["state"] == "done" and d["image_url"].startswith("/api/images/")
        img = await client.get(d["image_url"])
        assert img.status_code == 200 and img.headers["content-type"] == "image/png"
        gal = (await client.get("/api/images/gallery")).json()["images"]
        assert gal[0]["prompt"] == "a fox"
        assert (await client.get("/api/images/jobs/nope")).status_code == 404


async def test_generate_refuses_when_not_set_up(image_setup):
    (image_setup / "sd-cli").unlink()
    async with _client() as client:
        r = await client.post("/api/images/generate", json={"prompt": "x"})
        assert r.status_code == 503 and "sd-cli" in r.json()["detail"]


def test_text_encoder_found_by_folder_whatever_its_name(image_setup):
    (image_setup / "text_encoders" / "Qwen3VL-Q4_K_M.gguf").unlink()
    (image_setup / "text_encoders" / "qwen_2.5_vl_7b_fp8_scaled.safetensors").write_bytes(b"0")
    st = image_gen.status()
    assert st["ready"] is True and st["text_encoder"] == "qwen_2.5_vl_7b_fp8_scaled.safetensors"


def test_image_weights_never_appear_in_the_chat_picker(image_setup, tmp_path):
    import registry
    chat_dir = tmp_path / "Models" / "Chat"
    chat_dir.mkdir()
    (chat_dir / "Qwen3-4B-Q5_K_M.gguf").write_bytes(b"0")
    (tmp_path / "Models" / "qwen-image-2.1-turbo-Q4_K_M.gguf").write_bytes(b"0")  # stray, outside Image/
    registry._SCAN_CACHE.clear()
    ids = [m.id for m in registry.scan_models(tmp_path / "Models")]
    assert ids == ["qwen3-4b-q5-k-m"]
