"""Is HIVI getting the whole GPU?  Run with a model loaded for the full picture.

    .venv\\Scripts\\python.exe scripts\\gpu_check.py
    .venv\\Scripts\\python.exe scripts\\gpu_check.py --json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "interfaces" / "hivi-chat"))

import gpu_doctor  # noqa: E402

MARK = {"ok": "[ OK ]", "warn": "[WARN]", "bad": "[FIX!]"}


def main() -> int:
    rep = gpu_doctor.collect()
    if "--json" in sys.argv:
        print(json.dumps(rep.to_dict(), indent=2))
        return 0
    g = rep.gpu or {}
    print("HIVI GPU check")
    print(f"  llama-server : {rep.llama_server}")
    for d in rep.devices:
        print(f"  device       : {d['id']} {d['name']} ({d['free_mib']}/{d['total_mib']} MiB free)")
    if g:
        print(f"  GPU          : {g['name']}  {g['pstate']}  {g.get('temp_c') or 0:.0f} C")
        print(f"  VRAM         : {g.get('vram_used_mib') or 0:.0f} / {g.get('vram_total_mib') or 0:.0f} MiB used")
        print(f"  power        : {g.get('power_w') or 0:.0f} W now, limit {g.get('power_limit_w') or 0:.0f} W"
              f" (max {g.get('power_max_w') or 0:.0f} W)")
    if rep.battery:
        print(f"  battery      : {rep.battery['percent']:.0f}%  {'plugged in' if rep.battery['plugged'] else 'ON BATTERY'}")
    if rep.power_plan:
        print(f"  power mode   : {rep.power_plan}")
    for p in rep.processes:
        mib = f"{p['mib']} MiB" if p["mib"] is not None else "n/a"
        print(f"  on GPU       : {p['name']} ({p['type']}, {mib})")
    print()
    for f in rep.findings:
        print(f"{MARK[f['level']]} {f['text']}")
    return 1 if any(f["level"] == "bad" for f in rep.findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
