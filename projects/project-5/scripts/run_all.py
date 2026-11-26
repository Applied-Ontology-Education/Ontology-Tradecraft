"""Step 8: one-command driver for the whole Project 5 pipeline.

    python scripts/run_all.py

Runs every step in order and prints the summary: mean_cos, chosen tau, number of accepted axioms,
LLM contribution rate, and consistency status. LLM replies are cached in data/llm_cache.json, so a re-run
reproduces the same results without an API key; a first run needs ANTHROPIC_API_KEY."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STEPS = [
    ("extract_definitions.py", "data/definitions.csv"),
    ("preprocess_definitions_llm.py", "data/definitions_enriched.csv"),
    ("generate_candidates_llm.py", "generated/candidate_el.ttl"),
    ("split_module.py", "src/train.ttl + src/valid.ttl"),
    ("train_mowl.py", "reports/mowl_metrics.json"),
    ("filter_candidates_hybrid.py", "generated/accepted_el.ttl"),
    ("merge_and_reason.py", "src/module_augmented.ttl"),
    ("oops_check.py", "reports/oops_report.xml (optional QC, never fails)"),
]


def main():
    t0 = time.time()
    for i, (script, out) in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {script} -> {out}", flush=True)
        p = subprocess.run([sys.executable, str(HERE / script)], cwd=ROOT, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = [l for l in p.stdout.splitlines() if l.startswith(script.split(".")[0])]
        print("\n".join(lines) if lines else p.stdout[-2000:])
        if p.returncode != 0:
            print(p.stdout[-3000:])
            sys.exit(f"FAILED at step {i}: {script}")
    m = json.loads((ROOT / "reports" / "mowl_metrics.json").read_text())
    h, mg = m["hybrid"], m["merge"]
    print("\n================ Project 5 summary ================")
    print(f"mean_cos (held-out entailed pairs) : {m['mean_cos']:.3f}  (>= 0.70: {m['passes_mean_cos_0_70']})")
    print(f"  valid.ttl direct pairs           : {m['valid_direct']['mean']:.3f}   random baseline: {m['random_baseline']['mean']:.3f}")
    print(f"chosen tau                         : {m['tau']:.2f}")
    print(f"accepted axioms                    : {h['accepted']} of {h['candidates']} candidates "
          f"(subclass {h['accepted_by_pattern']['A_subclass']}, function {h['accepted_by_pattern']['B_function']})")
    print(f"LLM contribution rate              : {h['llm_contribution_rate']:.0%} of accepted needed the LLM score")
    print(f"weight sensitivity (diagnostic)    : {h['diagnostic_weight_sensitivity_not_applied']} candidates >= tau, before ELK")
    print(f"consistency (ELK)                  : {'CONSISTENT' if mg['consistent'] else 'INCONSISTENT'}; "
          f"accepted axioms OWL 2 EL: {mg['el_profile_accepted_axioms']}")
    print(f"OOPS! pitfall scan                 : {m.get('oops', {}).get('status', 'not run')}"
          + (f" ({', '.join(m['oops']['pitfall_codes']) or 'no pitfalls'})" if m.get('oops', {}).get('status') == 'ok' else ""))
    print(f"runtime                            : {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
