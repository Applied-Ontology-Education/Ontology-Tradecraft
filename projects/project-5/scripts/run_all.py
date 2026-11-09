"""One-command driver for the whole pipeline:  python scripts/run_all.py"""
import json
import os
import py_compile
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_DIR / "scripts"
STEPS = [
    ("1. Extract definitions", "extract_definitions.py"),
    ("2. Enrich definitions (LLM)", "preprocess_definitions_llm.py"),
    ("3. Generate candidate axioms (LLM)", "generate_candidates_llm.py"),
    ("4. Split train/valid", "split_ontology.py"),
    ("5. Train mOWL", "train_mowl.py"),
    ("6. Hybrid filter (LLM)", "filter_candidates_hybrid.py"),
    ("7. Merge and reason (ELK)", "merge_and_reason.py"),
]


def preflight():
    """Catch setup problems in seconds instead of halfway through the pipeline."""
    problems = []
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        problems.append(f"Python {sys.version.split()[0]}: run with .venv-mowl (mOWL needs 3.10-3.13)")
    if shutil.which("robot") is None:
        problems.append("robot not found on PATH")
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        problems.append("ANTHROPIC_API_KEY is not set (check .env)")
    for _, script in STEPS:
        try:
            py_compile.compile(str(SCRIPTS / script), doraise=True)
        except py_compile.PyCompileError as e:
            problems.append(f"syntax error in {script}: {e.msg}")
    jvm = subprocess.run([sys.executable, "-c", "import mowl; mowl.init_jvm('2g')"],
                         capture_output=True, text=True)
    if jvm.returncode != 0:
        problems.append("mOWL could not start the JVM (check JAVA_HOME)")
    if problems:
        sys.exit("Preflight failed:\n  - " + "\n  - ".join(problems))
    print("Preflight OK")


def run_step(name, script):
    print(f"\n=== {name} ===", flush=True)
    start = time.time()
    result = subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=PROJECT_DIR)
    if result.returncode != 0:
        sys.exit(f"\nFAILED at '{name}' (exit code {result.returncode}). "
                 f"Fix it, then rerun scripts/{script} on its own.")
    print(f"--- {name}: done in {time.time() - start:.0f}s")


def summary():
    with open(PROJECT_DIR / "config.yaml", encoding="utf-8") as f:
        paths = yaml.safe_load(f)["paths"]
    m = json.loads((PROJECT_DIR / paths["mowl_metrics"]).read_text(encoding="utf-8"))
    h = m["hybrid"]
    n_above = round(m["coverage_at_tau"] * m["n_valid_pairs"])
    print("\n=== Summary ===")
    print(f"mean_cos, all {m['n_valid_pairs']} held-out pairs: {m['mean_cos_all']:.3f}  "
          f"(random baseline {m['baseline_mean_cos']:.3f})")
    print(f"mean_cos, pairs with cos >= tau:   {m['mean_cos_at_tau']:.3f}  "
          f"({n_above} of {m['n_valid_pairs']} pairs, {m['coverage_at_tau']:.1%})")
    print(f"chosen tau:                        {m['chosen_tau']:.2f}")
    print(f"accepted axioms:                   {h['n_accepted']} of {h['n_candidates']} candidates")
    print(f"LLM contribution rate:             {h['llm_contribution_rate']:.1%}")
    print(f"consistency (ELK):                 {'consistent' if m.get('consistent') else 'NOT consistent'}")


def main():
    preflight()
    for name, script in STEPS:
        run_step(name, script)
    summary()


if __name__ == "__main__":
    main()
