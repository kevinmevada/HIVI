"""Run labeled-set evaluation against live HAIE plan_request (no routing mutation)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from haie import plan_request
from haie.policy import derive_params
from haie.types import Understanding

from .metrics import compute_classification_metrics

from hivi_paths import paths  # noqa: E402

ROOT = paths.repo_root()
BASELINE_PATH = ROOT / "data" / "haie" / "V1.2_P0_BASELINE.json"
LABELED_PATH = ROOT / "data" / "haie" / "p1" / "labeled_set.jsonl"
REPORT_DIR = ROOT / "data" / "haie" / "p1" / "reports"

# Proposed P1 targets (not yet measured claims)
TARGETS = {
    "min_labeled": 200,
    "macro_f1": 0.90,
    "ece": 0.05,
    "smoke_acceptance": "8/8",
}


def baseline_fingerprint(path: Path = BASELINE_PATH) -> dict[str, Any]:
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "version": data.get("version"),
        "status": data.get("status"),
        "frozen_at": data.get("frozen_at"),
        "acceptance": data.get("acceptance"),
        "immutable": True,
    }


def load_labeled_set(path: Path = LABELED_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def stratified_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Take roughly equal counts per split so --limit is not all-simple prefix."""
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    by_split: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_split.setdefault(str(r.get("split") or "other"), []).append(r)
    splits = sorted(by_split.keys())
    if not splits:
        return rows[:limit]
    out: list[dict[str, Any]] = []
    # Round-robin across splits
    idxs = {s: 0 for s in splits}
    while len(out) < limit:
        progressed = False
        for s in splits:
            i = idxs[s]
            if i < len(by_split[s]):
                out.append(by_split[s][i])
                idxs[s] = i + 1
                progressed = True
                if len(out) >= limit:
                    break
        if not progressed:
            break
    return out


def _gold_understanding(row: dict[str, Any]) -> Understanding:
    return Understanding(
        intents=list(row["gold_intents"]),
        complexity=int(row.get("gold_complexity") or 2),
        deliverable=str(row.get("gold_deliverable") or "reply"),
        reasoning=str(row.get("gold_reasoning") or "low"),
        length=str(row.get("gold_length") or "medium"),
        ambiguity=float(row.get("gold_ambiguity") or 0.3),
        confidence=1.0,
        path=str(row.get("gold_path") or "semantic"),
        expected_output_tokens=row.get("gold_expected_tokens"),
    ).normalized()


def _profile_bucket(params: dict[str, Any]) -> tuple[float, int]:
    """Coarse profile for mismatch: temp to 0.05, max_tokens to 256."""
    temp = round(float(params.get("temperature") or 0.0) / 0.05) * 0.05
    mx = int(params.get("max_tokens") or 0)
    mx_b = max(256, int(round(mx / 256.0) * 256))
    return (round(temp, 2), mx_b)


def evaluate_labeled_set(
    *,
    rows: list[dict[str, Any]],
    backend_base: str,
    engine: str | None,
    model_id: str | None,
    limit: int | None = None,
    stratified: bool = True,
) -> dict[str, Any]:
    if limit is None:
        subset = list(rows)
    elif stratified:
        subset = stratified_sample(rows, limit)
    else:
        subset = rows[:limit]
    gold_intents: list[list[str]] = []
    pred_intents: list[list[str]] = []
    gold_paths: list[str] = []
    pred_paths: list[str] = []
    confidences: list[float] = []
    param_mismatch: list[bool] = []
    classify_attempted: list[bool] = []
    details: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    for row in subset:
        prompt = row["prompt"]
        body = {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        adapted, meta = plan_request(
            body,
            engine=engine,
            backend_base=backend_base,
            model_id=model_id,
        )
        gold_u = _gold_understanding(row)
        gold_params = derive_params(gold_u)
        gold_bucket = _profile_bucket(
            {
                "temperature": gold_params.temperature,
                "max_tokens": gold_params.max_tokens,
            }
        )
        pred_bucket = _profile_bucket(
            {
                "temperature": adapted.get("temperature"),
                "max_tokens": adapted.get("max_tokens"),
            }
        )
        mismatch = gold_bucket != pred_bucket
        # Wrong intents that still land in same coarse profile are tracked separately
        intents_exact = set(row["gold_intents"]) == set(meta.intents)
        routing_impact = (not intents_exact) and mismatch

        gold_intents.append(list(row["gold_intents"]))
        pred_intents.append(list(meta.intents))
        gold_paths.append(str(row["gold_path"]))
        pred_paths.append(str(meta.path))
        confidences.append(float(meta.confidence))
        param_mismatch.append(routing_impact)
        classify_attempted.append(meta.path != "fast" or meta.classify_ms is not None)

        details.append(
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "prompt": prompt,
                "gold_intents": row["gold_intents"],
                "pred_intents": meta.intents,
                "gold_path": row["gold_path"],
                "pred_path": meta.path,
                "confidence": meta.confidence,
                "exact_intents": intents_exact,
                "path_ok": row["gold_path"] == meta.path,
                "routing_impact": routing_impact,
                "classify_ms": meta.classify_ms,
                "pred_temp": adapted.get("temperature"),
                "pred_max_tokens": adapted.get("max_tokens"),
                "gold_temp": gold_params.temperature,
                "gold_max_tokens": gold_params.max_tokens,
            }
        )

    metrics = compute_classification_metrics(
        gold_intents=gold_intents,
        pred_intents=pred_intents,
        gold_paths=gold_paths,
        pred_paths=pred_paths,
        confidences=confidences,
        param_mismatch=param_mismatch,
        classify_attempted=classify_attempted,
    )
    elapsed_s = time.perf_counter() - t0

    targets_status = {
        "min_labeled": {
            "target": TARGETS["min_labeled"],
            "actual": len(rows),
            "pass": len(rows) >= TARGETS["min_labeled"],
        },
        "macro_f1": {
            "target": TARGETS["macro_f1"],
            "actual": round(metrics.macro_f1, 4),
            "pass": metrics.macro_f1 >= TARGETS["macro_f1"],
        },
        "ece": {
            "target": TARGETS["ece"],
            "actual": round(metrics.ece, 4),
            "pass": metrics.ece <= TARGETS["ece"],
        },
    }

    return {
        "baseline": baseline_fingerprint(),
        "dataset": {
            "path": str(LABELED_PATH),
            "total_labeled": len(rows),
            "evaluated": len(subset),
            "limit": limit,
        },
        "runtime": {
            "engine": engine,
            "model_id": model_id,
            "backend_base": backend_base,
            "elapsed_s": round(elapsed_s, 2),
        },
        "targets": TARGETS,
        "targets_status": targets_status,
        "metrics": metrics.as_dict(),
        "split_counts": _split_counts(subset),
        "details": details,
        "note": (
            "Measurement only — production routing unchanged. "
            "V1.2_P0_BASELINE.json is read-only (fingerprint recorded)."
        ),
    }


def _split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        s = str(r.get("split") or "other")
        out[s] = out.get(s, 0) + 1
    return out


def write_report(report: dict[str, Any], path: Path | None = None) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = REPORT_DIR / f"calibration_{stamp}.json"
    # Full report with details
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Slim summary alongside
    slim = {k: report[k] for k in report if k != "details"}
    slim_path = path.with_name(path.stem + "_summary.json")
    slim_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    latest = REPORT_DIR / "latest_summary.json"
    latest.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return path
