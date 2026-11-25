import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
REPORTS_DIR = PROJECT_DIR / "reports"

STEPS = [
    ("Extract definitions", "extract_definitions.py"),
    ("Normalize definitions (LLM)", "preprocess_definitions_llm.py"),
    ("Generate candidate axioms (LLM)", "generate_candidates_llm.py"),
    ("Split train / validation", "split_ontology.py"),
    ("Train MOWL ELEmbeddings", "train_mowl.py"),
    ("Hybrid filter (MOWL + LLM)", "filter_candidates_hybrid.py"),
    ("Merge + ELK reasoning", "merge_and_reason.py"),
]


def load_env():
    """Load GEMINI_API_KEY from a .env file if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (PROJECT_DIR / ".env", PROJECT_DIR.parents[1] / ".env"):
        if candidate.exists():
            load_dotenv(candidate)


def run_step(number, title, script):
    path = SCRIPTS_DIR / script
    print(f"\n{'=' * 60}\nSTEP {number}/{len(STEPS)}: {title}  ({script})\n{'=' * 60}")

    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)

    start = time.time()
    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(PROJECT_DIR),
        env=os.environ.copy(),
    )
    elapsed = time.time() - start

    if completed.returncode != 0:
        print(f"\nPIPELINE STOPPED: step {number} ({script}) failed "
              f"with exit code {completed.returncode}.")
        sys.exit(completed.returncode)

    print(f"\n[step {number} finished in {elapsed:.1f}s]")


def read_json(name):
    path = REPORTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Expected report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize():
    mowl = read_json("mowl_metrics.json")["metrics"]
    hybrid = read_json("hybrid_filter_report.json")
    merge = read_json("merge_report.json")

    contribution = hybrid["llm_contribution"]

    summary = {
        "mean_cos": mowl["validation_mean_cosine"],
        "mean_cos_target": mowl.get("target_mean_cosine", 0.70),
        "random_control_mean_cos": mowl.get("random_control_mean_cosine"),
        "chosen_tau": mowl["selected_threshold"],
        "candidates_scored": hybrid["candidate_count"],
        "accepted_axioms": hybrid["accepted_count"],
        "new_axioms_added": merge["new_axioms_added"],
        "elk_inferred_axioms": merge["inferred_subclass_axioms_added_by_elk"],
        "llm_contribution_rate": contribution["llm_contribution_rate"],
        "llm_decisions_changed": contribution["decisions_changed_by_llm"],
        "mean_llm_share_of_hybrid_score": contribution["mean_llm_share_of_hybrid_score"],
        "consistent": merge["consistent"],
    }

    (REPORTS_DIR / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    passed = summary["mean_cos"] >= summary["mean_cos_target"]

    print(f"\n{'=' * 60}\nPIPELINE SUMMARY\n{'=' * 60}")
    print(f"mean_cos (validation):     {summary['mean_cos']:.3f}  "
          f"({'PASS' if passed else 'FAIL'}, target >= {summary['mean_cos_target']:.2f}; "
          f"random pairs {summary['random_control_mean_cos']:.3f})")
    print(f"Chosen tau:                {summary['chosen_tau']:.2f}")
    print(f"Accepted axioms:           {summary['accepted_axioms']} of "
          f"{summary['candidates_scored']} scored candidates")
    print(f"New axioms in module:      {summary['new_axioms_added']} "
          f"(+{summary['elk_inferred_axioms']} inferred by ELK)")
    print(f"LLM contribution rate:     {summary['llm_contribution_rate']:.1%} "
          f"({summary['llm_decisions_changed']} decisions changed; "
          f"LLM supplies {summary['mean_llm_share_of_hybrid_score']:.1%} of hybrid score)")
    print(f"Consistency (ELK):         "
          f"{'CONSISTENT' if summary['consistent'] else 'INCONSISTENT'}")
    print(f"\nSummary saved to {REPORTS_DIR / 'pipeline_summary.json'}")


def main():
    load_env()
    total_start = time.time()

    for number, (title, script) in enumerate(STEPS, start=1):
        run_step(number, title, script)

    summarize()
    print(f"Total runtime: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()