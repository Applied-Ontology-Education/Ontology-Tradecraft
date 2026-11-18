"""Audit Project 6 Step 5's raw, independent cosine requirement.

This is a READ-ONLY audit of existing scores. It does not retrain mOWL,
change ontology files, or convert Grok plausibility into cosine evidence.

Usage:
  python scripts/audit_step5_cosine.py --project-root .
  python scripts/audit_step5_cosine.py --project-root . --threshold 0.70 --strict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

VERSION = "project6-step5-cosine-requirement-audit-v1"
RAIL_SCOPE = "RAIL_NEW_DOMAIN"
REF_SCOPE = "SOURCE_IMPORT_REFERENCE"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean_or_none(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def summarize(scores: dict, *, threshold: float = 0.70,
              min_independent_rail_pairs: int = 1) -> dict:
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Cosine threshold must be finite and within [-1, 1]")
    if min_independent_rail_pairs < 1:
        raise ValueError("min_independent_rail_pairs must be >= 1")
    rows = scores.get("scores")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing or empty scores array")
    scoped: dict[str, list[float]] = {RAIL_SCOPE: [], REF_SCOPE: []}
    not_scored: dict[str, int] = {RAIL_SCOPE: 0, REF_SCOPE: 0}
    edges: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed score row")
        scope = row.get("scope")
        if scope not in scoped:
            raise ValueError(f"Unexpected score scope: {scope}")
        child, parent = row.get("child"), row.get("parent")
        if not isinstance(child, str) or not isinstance(parent, str):
            raise ValueError("Score row is missing class IRIs")
        pair = (child, parent)
        if pair in edges:
            raise ValueError(f"Duplicate scored subclass pair: {pair}")
        edges.add(pair)
        value = row.get("cosine")
        if value is None:
            not_scored[scope] += 1
            continue
        if (row.get("status") != "HELD_OUT_COSINE" or
                isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not -1 <= value <= 1):
            raise ValueError(f"Invalid independent held-out cosine for {pair}")
        scoped[scope].append(float(value))
    rail = scoped[RAIL_SCOPE]
    ref = scoped[REF_SCOPE]
    all_values = rail + ref
    if scores.get("heldout_source_import_pairs") != len(ref):
        raise ValueError("Reference sample size does not match original report")
    reported_rail = scores.get("heldout_rail_cosine")
    if reported_rail is not None and len(rail) == 1 and abs(rail[0] - reported_rail) > 1e-6:
        raise ValueError("Rail cosine disagrees with original report")
    if scores.get("mean_heldout_source_import_cosine") is not None and ref:
        if abs(statistics.mean(ref) - scores["mean_heldout_source_import_cosine"]) > 1e-5:
            raise ValueError("Reference cosine mean disagrees with original report")
    rail_mean = mean_or_none(rail)
    meets_threshold = (len(rail) >= min_independent_rail_pairs and
                       rail_mean is not None and rail_mean >= threshold)
    return {
        "version": VERSION,
        "status": "PASS_COSINE_REQUIREMENT" if meets_threshold else "FAIL_COSINE_REQUIREMENT",
        "metric": "arithmetic mean of raw mOWL class-center cosine similarity on independently held-out new rail subclass pairs",
        "threshold": threshold,
        "threshold_met": meets_threshold,
        "minimum_independent_rail_pairs": min_independent_rail_pairs,
        "rail_heldout_pairs": len(rail),
        "rail_unscored_pairs": not_scored[RAIL_SCOPE],
        "rail_heldout_mean_cosine": rail_mean,
        "rail_at_or_above_threshold": sum(x >= threshold for x in rail),
        "reference_heldout_pairs": len(ref),
        "reference_heldout_mean_cosine": mean_or_none(ref),
        "reference_at_or_above_threshold": sum(x >= threshold for x in ref),
        "all_heldout_pairs": len(all_values),
        "all_heldout_mean_cosine": mean_or_none(all_values),
        "highest_observed_cosine": round(max(all_values), 6) if all_values else None,
        "reference_comparison_is_not_a_rail_candidate_average": True,
        "hybrid_weight_0_70_is_not_a_cosine_threshold": True,
        "small_sample_warning": len(rail) < 5,
        "note": "This reports a threshold; it does not change model training or improve scores. Cosine is not the ELEmbeddings native ball-inclusion loss.",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--scores", type=Path, default=None, help="Override source score report path")
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--min-independent-rail-pairs", type=int, default=1)
    p.add_argument("--strict", action="store_true", help="Return nonzero exit status if threshold not met")
    p.add_argument("--no-write", action="store_true", help="Do not write audit JSON")
    args = p.parse_args()
    root = args.project_root.resolve()
    path = args.scores or (root / "reports" / "step5_cosine_scores.json")
    try:
        scores = json.loads(path.read_text(encoding="utf-8"))
        report = summarize(scores, threshold=args.threshold,
                           min_independent_rail_pairs=args.min_independent_rail_pairs)
        report["input_scores_sha256"] = sha256(path)
        if not args.no_write:
            output = root / "reports" / "step5_cosine_requirement_audit.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("STEP 5 COSINE AUDIT - " + report["status"])
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2 if args.strict and not report["threshold_met"] else 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"STEP 5 COSINE AUDIT - ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
