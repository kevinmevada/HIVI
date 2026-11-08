"""Classification + calibration metrics for HAIE P1 (read-only measurement)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence


VALID_INTENTS = (
    "greeting",
    "coding",
    "explanation",
    "architecture",
    "math",
    "research",
    "brainstorm",
    "chat",
)


@dataclass
class ClassificationMetrics:
    n: int
    exact_intent_accuracy: float
    primary_intent_accuracy: float
    path_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_intent: dict[str, dict[str, float]] = field(default_factory=dict)
    multi_intent_f1: float = 0.0
    ece: float = 0.0
    ece_primary: float = 0.0
    ece_bins: list[dict[str, Any]] = field(default_factory=list)
    fallback_rate: float = 0.0
    classify_failure_rate: float = 0.0  # path==fallback among expected-semantic
    invalid_json_proxy_rate: float = 0.0  # same signal today: fallback after semantic attempt
    param_profile_mismatch_rate: float = 0.0
    mean_confidence: float = 0.0
    mean_confidence_when_correct: float = 0.0
    mean_confidence_when_wrong: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def multi_label_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    g, p = set(gold), set(pred)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    tp = len(g & p)
    precision = _safe_div(tp, len(p))
    recall = _safe_div(tp, len(g))
    return _safe_div(2 * precision * recall, precision + recall)


def compute_ece(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    """Expected Calibration Error over equal-width confidence bins."""
    if not confidences:
        return 0.0, []
    n = len(confidences)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idx = [
            i
            for i, c in enumerate(confidences)
            if (c >= lo and c < hi) or (b == n_bins - 1 and c == 1.0)
        ]
        if not idx:
            bins.append(
                {
                    "bin": b,
                    "lo": lo,
                    "hi": hi,
                    "count": 0,
                    "confidence": None,
                    "accuracy": None,
                    "gap": None,
                }
            )
            continue
        conf_avg = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        gap = abs(conf_avg - acc)
        ece += (len(idx) / n) * gap
        bins.append(
            {
                "bin": b,
                "lo": round(lo, 3),
                "hi": round(hi, 3),
                "count": len(idx),
                "confidence": round(conf_avg, 4),
                "accuracy": round(acc, 4),
                "gap": round(gap, 4),
            }
        )
    return ece, bins


def reliability_bins(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> list[dict[str, Any]]:
    _, bins = compute_ece(confidences, correct, n_bins=n_bins)
    return bins


def compute_classification_metrics(
    *,
    gold_intents: Sequence[Sequence[str]],
    pred_intents: Sequence[Sequence[str]],
    gold_paths: Sequence[str],
    pred_paths: Sequence[str],
    confidences: Sequence[float],
    param_mismatch: Sequence[bool],
    classify_attempted: Sequence[bool] | None = None,
) -> ClassificationMetrics:
    n = len(gold_intents)
    assert n == len(pred_intents) == len(gold_paths) == len(pred_paths) == len(confidences)

    exact = [
        set(g) == set(p) for g, p in zip(gold_intents, pred_intents, strict=True)
    ]
    primary_ok = [
        (g[0] if g else "chat") == (p[0] if p else "chat")
        for g, p in zip(gold_intents, pred_intents, strict=True)
    ]
    path_ok = [g == p for g, p in zip(gold_paths, pred_paths, strict=True)]

    # Per-intent binary metrics (multi-label)
    per_intent: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    precs: list[float] = []
    recs: list[float] = []
    for intent in VALID_INTENTS:
        tp = fp = fn = 0
        for g, p in zip(gold_intents, pred_intents, strict=True):
            gin, pin = intent in g, intent in p
            if gin and pin:
                tp += 1
            elif pin and not gin:
                fp += 1
            elif gin and not pin:
                fn += 1
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec)
        support = sum(1 for g in gold_intents if intent in g)
        per_intent[intent] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": float(support),
        }
        if support > 0:
            f1s.append(f1)
            precs.append(prec)
            recs.append(rec)

    ml_f1 = [
        multi_label_f1(g, p) for g, p in zip(gold_intents, pred_intents, strict=True)
    ]

    ece, bins = compute_ece(confidences, exact, n_bins=10)
    ece_primary, _ = compute_ece(confidences, primary_ok, n_bins=10)

    fallback = sum(1 for p in pred_paths if p == "fallback")
    if classify_attempted is None:
        # heuristic: expected semantic/fast mismatch into fallback
        classify_attempted = [gp != "fast" or pp == "fallback" for gp, pp in zip(gold_paths, pred_paths)]
    semantic_expected = [i for i, gp in enumerate(gold_paths) if gp == "semantic"]
    fail_among_semantic = sum(
        1 for i in semantic_expected if pred_paths[i] == "fallback"
    )

    correct_conf = [c for c, ok in zip(confidences, exact) if ok]
    wrong_conf = [c for c, ok in zip(confidences, exact) if not ok]

    return ClassificationMetrics(
        n=n,
        exact_intent_accuracy=_safe_div(sum(exact), n),
        primary_intent_accuracy=_safe_div(sum(primary_ok), n),
        path_accuracy=_safe_div(sum(path_ok), n),
        macro_precision=_safe_div(sum(precs), len(precs)),
        macro_recall=_safe_div(sum(recs), len(recs)),
        macro_f1=_safe_div(sum(f1s), len(f1s)),
        per_intent=per_intent,
        multi_intent_f1=_safe_div(sum(ml_f1), n),
        ece=ece,
        ece_primary=ece_primary,
        ece_bins=bins,
        fallback_rate=_safe_div(fallback, n),
        classify_failure_rate=_safe_div(fail_among_semantic, len(semantic_expected)),
        invalid_json_proxy_rate=_safe_div(fail_among_semantic, len(semantic_expected)),
        param_profile_mismatch_rate=_safe_div(sum(param_mismatch), n),
        mean_confidence=_safe_div(sum(confidences), n),
        mean_confidence_when_correct=_safe_div(sum(correct_conf), len(correct_conf)),
        mean_confidence_when_wrong=_safe_div(sum(wrong_conf), len(wrong_conf)),
    )
