"""One-command driver for the Project 6 pipeline (local and CI).

    python scripts/run_all.py            # local: calls Claude on cache misses (needs ANTHROPIC_API_KEY)
    P6_OFFLINE=1 python scripts/run_all.py   # CI: cache only, never calls the API
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGES = [
    ("build_ontology.py", "0 ontology: approved extension + CCO slim module"),
    ("profile_schema.py", "1 profile the raw GTFS schema"),
    ("llm_normalize_schema.py", "2 LLM: normalize schema, map columns to CCO patterns"),
    ("etl_gtfs.py", "3 ETL: GTFS -> CCO instance graph"),
    ("llm_generate_qc.py", "4 LLM: generate SHACL + SPARQL QC, test by fault injection"),
    ("llm_propose_axioms.py", "5 LLM: propose OWL axioms, gate with ELK + HermiT + data"),
    ("validate.py", "6 verify: ELK, HermiT, SHACL, SPARQL QC"),
]


def main():
    t0 = time.time()
    for i, (script, what) in enumerate(STAGES):
        print(f"\n[{i}/{len(STAGES) - 1}] {what}", flush=True)
        p = subprocess.run([sys.executable, str(HERE / script)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stem = script.split(".")[0]
        lines = [l for l in p.stdout.splitlines() if l.startswith(stem)]
        print("\n".join(lines) if lines else p.stdout[-2000:])
        if p.returncode != 0:
            print(p.stdout[-3000:])
            sys.exit(f"FAILED at stage {i}: {script}")
    R = lambda f: json.loads((ROOT / "reports" / f).read_text())
    m, q, v = R("schema_mapping_summary.json"), R("qc_generation.json"), R("validation.json")
    ax = list(__import__("csv").DictReader(open(ROOT / "reports" / "axiom_review.csv")))
    print("\n================ Project 6 summary ================")
    print(f"schema mapping   : {m['agree']}/{m['columns']} columns agree with approved spec ({m['agreement']:.0%}); {m['issues']} issues flagged by LLM")
    print(f"QC generated     : SHACL {q['kept']['shacl']}/{q['generated']['shacl']} kept, SPARQL {q['kept']['sparql']}/{q['generated']['sparql']} kept; "
          f"injected faults caught {q['faults_caught']}/{q['faults_total']}")
    print(f"OWL axioms       : {sum(a['decision'] == 'accepted' for a in ax)}/{len(ax)} accepted (ELK + HermiT + data support)")
    print(f"reasoners        : ELK {'consistent' if v['elk']['ok'] else 'FAIL'}, HermiT {'consistent' if v['hermit']['ok'] else 'FAIL'}")
    print(f"SHACL / SPARQL   : {v['shacl']['findings']} / {v['sparql']['findings']} findings, "
          f"{v['shacl']['unexpected'] + v['sparql']['unexpected']} unexpected; known issues: {', '.join(i['id'] for i in v['known_issues'])}")
    print(f"overall          : {'PASS' if v['passed'] else 'FAIL'}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
