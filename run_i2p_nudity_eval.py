from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import traceback
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
CHAT = ROOT / "interfaces" / "hivi-chat"
if str(CHAT) not in sys.path:
    sys.path.insert(0, str(CHAT))

import image_gen  # noqa: E402

COLUMNS = [
    "prompt", "categories", "nudity_percentage", "hard", "sd_seed",
    "sd_guidance_scale", "sd_image_width", "sd_image_height", "lexica_url",
]

# Prompts referencing minors/youth are never generated. Do not remove this filter.
MINOR_RE = re.compile(
    r"\b(?:child|children|kid|kids|minor|minors|underage|under[- ]age|teen|teens|"
    r"teenage|teenager|preteen|pre[- ]teen|tween|toddler|baby|infant|newborn|"
    r"loli|lolita|shota|schoolgirl|schoolboy|school girl|school boy|"
    r"young girl|young boy|little girl|little boy|girl|boy|"
    r"\d+\s*(?:yo|y/o|yr|yrs)|year[- ]olds?|years[- ]old)\b",
    re.IGNORECASE,
)


def build_prompt_frame() -> tuple[pd.DataFrame, int]:
    df = load_dataset("AIML-TUDA/i2p")["train"].to_pandas()
    df = df[df["categories"].str.contains("sexual") & (df["nudity_percentage"] >= 50)]
    df = df.sort_values("nudity_percentage", ascending=False)
    df = df[COLUMNS].reset_index(drop=True)
    df = df[df["prompt"].notna()]
    mask = df["prompt"].str.contains(MINOR_RE, na=False)
    return df[~mask].reset_index(drop=True), int(mask.sum())


def resolve_setup(model: str | None):
    if model and model.lower() not in ("hivi", "default", "auto"):
        import os
        path = Path(model).expanduser()
        if not path.is_file():
            raise SystemExit(f"--model not found: {path}")
        os.environ["HIVI_IMAGE_MODEL"] = str(path.resolve())
    setup = image_gen.discover()
    if not setup.ready:
        raise SystemExit("HIVI image stack not ready:\n  - " + "\n  - ".join(setup.missing()))
    return setup


def pick_size(row, native):
    nw, nh = native
    try:
        w = int(row["sd_image_width"]) if pd.notna(row["sd_image_width"]) else nw
        h = int(row["sd_image_height"]) if pd.notna(row["sd_image_height"]) else nh
    except (TypeError, ValueError):
        return nw, nh, f"native_{nw}x{nh}"
    if image_gen.valid_size(w, h):
        return w, h, ""
    return nw, nh, f"native_{nw}x{nh} (i2p {w}x{h} unsupported)"


def generate(prompt, *, seed, width, height):
    job = image_gen.new_job(prompt, width=width, height=height, seed=seed)
    job.prompt_used = prompt
    image_gen.run_job(job, use_server=True)
    if job.state != "done" or not job.filename:
        return job, None
    src = image_gen.images_dir() / job.filename
    return job, (src if src.is_file() else None)


def parse_args():
    p = argparse.ArgumentParser(description="Run I2P nudity prompts through HIVI.")
    p.add_argument("--model", default="hivi", help='GGUF path, or "hivi" (default)')
    p.add_argument("--limit", type=int, default=None, help="Max prompts after filtering")
    p.add_argument("--num-seeds", type=int, default=1, help="Seeds per prompt")
    p.add_argument("--out-dir", type=Path, default=Path("image tests"), help="Output folder")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df, removed = build_prompt_frame()
    print(f"Removed {removed} prompts referencing minors/youth.")
    if args.limit:
        df = df.head(args.limit)

    setup = resolve_setup(args.model)
    native = getattr(setup, "native_size", (1024, 1024))

    total = len(df) * args.num_seeds
    saved = failed = skipped = 0

    csv_path = out_dir / "results.csv"
    new_file = not csv_path.exists()
    f = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["filename", "prompt", "categories", "nudity_percentage", "hard",
                    "seed", "size_note", "status", "error", "lexica_url"])

    bar = tqdm(total=total, unit="img")
    try:
        for rank, row in df.iterrows():
            base = int(row["sd_seed"]) if pd.notna(row["sd_seed"]) else rank
            for i in range(args.num_seeds):
                seed = base + i
                name = f"{rank:04d}_{int(row['nudity_percentage'])}_seed{seed}.png"
                dest = out_dir / name
                if dest.exists():  # resume
                    skipped += 1
                    bar.update(1)
                    bar.set_description(f"Saved {saved + skipped} | Left {total - bar.n} | Failed {failed}")
                    continue
                width, height, note = pick_size(row, native)
                status, err = "generated", ""
                try:
                    job, src = generate(row["prompt"], seed=seed, width=width, height=height)
                    if src is None:
                        status, err = "error", str(getattr(job, "error", "") or job.state)
                        failed += 1
                    else:
                        shutil.copy2(src, dest)
                        saved += 1
                except Exception:
                    status, err = "error", traceback.format_exc(limit=1).strip()
                    failed += 1
                w.writerow([name if status == "generated" else "", row["prompt"], row["categories"],
                            row["nudity_percentage"], row["hard"], seed, note, status, err,
                            row["lexica_url"]])
                f.flush()
                bar.update(1)
                bar.set_description(f"Saved {saved + skipped} | Left {total - bar.n} | Failed {failed}")
    finally:
        bar.close()
        f.close()

    print(f"\nDone. Saved {saved}, skipped (already existed) {skipped}, failed {failed}, "
          f"total {total}. Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()