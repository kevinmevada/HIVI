"""GPU check: parsing real llama-server / nvidia-smi output and the advice it gives."""
from __future__ import annotations

import gpu_doctor as g
import runtime

LIST_CUDA = """ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5060 Laptop GPU, compute capability 12.0, VMM: yes
Available devices:
  CUDA0: NVIDIA GeForce RTX 5060 Laptop GPU (8151 MiB, 7362 MiB free)
"""
LIST_VULKAN_IGPU = """Available devices:
  Vulkan0: Intel(R) Arc(TM) 140T GPU (16384 MiB, 15000 MiB free)
"""
LIST_CPU_ONLY = "Available devices:\n"

QUERY_OK = ("NVIDIA GeForce RTX 5060 Laptop GPU, 8151, 7650, 97, 92.31, 115.00, 115.00, "
            "P0, 2610, 3090, 74, 0x0000000000000000\n")
QUERY_THROTTLED = ("NVIDIA GeForce RTX 5060 Laptop GPU, 8151, 3900, 40, 35.10, 60.00, 115.00, "
                   "P2, 1200, 3090, 88, 0x0000000000000084\n")

SMI_TABLE = """+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           12345      C   ...\\llama.cpp\\bin\\llama-server.exe    6789MiB |
|    0   N/A  N/A            4321    C+G   ...Application\\msedgewebview2.exe          N/A |
|    0   N/A  N/A            7777    C+G   ...\\Discord\\app-1.0\\Discord.exe          312MiB |
+-----------------------------------------------------------------------------------------+
"""


def _rep(devices=LIST_CUDA, query=QUERY_OK, procs=SMI_TABLE, battery=None, plan="Best performance"):
    rep = g.Report(llama_server="llama-server.exe")
    rep.devices = g.parse_list_devices(devices)
    rep.gpu = g.parse_gpu_query(query)
    rep.processes = g.parse_gpu_processes(procs)
    rep.battery = battery or {"percent": 90.0, "plugged": True}
    rep.power_plan = plan
    return rep


def _levels(findings, level):
    return [f["text"] for f in findings if f["level"] == level]


def test_parses_cuda_device():
    (d,) = g.parse_list_devices(LIST_CUDA)
    assert d["backend"] == "CUDA" and d["total_mib"] == 8151 and "RTX 5060" in d["name"]


def test_parses_gpu_query_and_processes():
    q = g.parse_gpu_query(QUERY_OK)
    assert q["vram_used_mib"] == 7650 and q["pstate"] == "P0" and q["power_limit_w"] == 115
    procs = g.parse_gpu_processes(SMI_TABLE)
    assert [p["name"] for p in procs] == ["llama-server.exe", "msedgewebview2.exe", "Discord.exe"]
    assert procs[0]["mib"] == 6789 and procs[1]["mib"] is None


def test_throttle_bits():
    assert g.throttle_reasons("0x0000000000000084") == ["power cap", "power brake (charger / battery)"]
    assert g.throttle_reasons("0x0") == [] and g.throttle_reasons(None) == []


def test_healthy_laptop_only_flags_other_gpu_apps():
    f = g.diagnose(_rep())
    assert not _levels(f, "bad")
    assert any("CUDA on NVIDIA GeForce RTX 5060" in t for t in _levels(f, "ok"))
    (warn,) = _levels(f, "warn")
    assert "Discord.exe" in warn and "msedgewebview2.exe" in warn and "llama-server" not in warn


def test_wrong_builds_are_called_out():
    for listing in (LIST_VULKAN_IGPU, LIST_CPU_ONLY):
        bad = _levels(g.diagnose(_rep(devices=listing)), "bad")
        assert bad and "CUDA" in bad[0]
    assert "Intel" in _levels(g.diagnose(_rep(devices=LIST_VULKAN_IGPU)), "bad")[0]


def test_battery_power_cap_heat_and_unused_vram():
    f = g.diagnose(_rep(query=QUERY_THROTTLED, battery={"percent": 41.0, "plugged": False},
                        plan="Balanced"))
    text = " | ".join(x["text"] for x in f)
    assert "On battery" in text and "60 W of a possible 115 W" in text
    assert "power brake" in text and "88 °C" in text and "'Balanced'" in text
    assert "4251 MiB of VRAM is unused" in text and "HIVI_FIT_TARGET_MIB" in text


def test_no_driver():
    rep = _rep()
    rep.gpu = None
    assert "nvidia-smi" in _levels(g.diagnose(rep), "bad")[0]


def test_power_mode_overlay_names():
    assert g.power_mode_name("DED574B5-45A0-4F42-8737-46345C09C238") == "Best performance"
    assert g.power_mode_name("unknown") is None


def test_fit_target_setting(monkeypatch, tmp_path):
    model = type("M", (), {"path": str(tmp_path / "m.gguf"), "id": "m", "uncensored": False,
                           "vision_projector_path": None})()
    args = runtime.build_llama_args(model, llama=tmp_path / "llama-server", threads=8)
    assert args[args.index("--fit-target") + 1] == "512"
    monkeypatch.setenv("HIVI_FIT_TARGET_MIB", "256")
    args = runtime.build_llama_args(model, llama=tmp_path / "llama-server", threads=8)
    assert args[args.index("--fit-target") + 1] == "256"
    monkeypatch.setenv("HIVI_FIT_TARGET_MIB", "10")  # too small to be safe: ignored
    args = runtime.build_llama_args(model, llama=tmp_path / "llama-server", threads=8)
    assert args[args.index("--fit-target") + 1] == "512"
