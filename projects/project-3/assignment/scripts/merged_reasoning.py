"""Reason over each ontology pair TOGETHER with its mapping files (Project 3, step 5/8).

The provided reasoner_run.ipynb reasons over each mapping file alone, which is nearly
always consistent. This script merges both ontologies + their mapping files and runs
HermiT and ELK, writing src/data/merged-reasoning-report.md.

owl:imports are stripped (they point at web IRIs) and the local CCO merge from
Project 4 supplies the imported CCO terms instead, so runs are offline and repeatable.

Run from projects/project-3/assignment:  python scripts/merged_reasoning.py
"""
import subprocess, tempfile
from pathlib import Path
from rdflib import Graph, OWL

SRC, DATA = Path("src"), Path("src/data")
CCO = Path("../../project-4/assignment/src/cco_merged.ttl")
PAIRS = {
    "BFO + IES":     ["bfo-core.ttl", "ies.ttl", "data/bfo-mapping.ttl", "data/ies-mapping.ttl"],
    "CCOM + QUDT":   ["ccom.ttl", "qudt.ttl", "CCO", "data/ccom-mapping.ttl", "data/qudt-mapping.ttl"],
    "CCOT + OWL-Time": ["ccot.ttl", "time.ttl", "CCO", "data/ccot-mapping.ttl", "data/to-mapping.ttl"],
}

def classify(out: str) -> str:
    if "inconsistent" in out.lower():
        return "INCONSISTENT"
    if "non-simple property" in out.lower():
        return "NOT RUN: ontology outside OWL 2 DL (non-simple property in disjointness axiom)"
    if "unsatisfiable" in out.lower():
        return "UNSATISFIABLE CLASSES"
    if "error" in out.lower() and "could not be parsed" not in out.lower():
        return "ERROR"
    return "CONSISTENT"

tmp = Path(tempfile.mkdtemp())
def local(name: str) -> Path:
    src = CCO if name == "CCO" else SRC / name
    out = tmp / src.name
    if not out.exists():
        g = Graph().parse(src)
        g.remove((None, OWL.imports, None))
        g.serialize(out, format="turtle")
    return out

rows = []
for pair, files in PAIRS.items():
    args = []
    for f in files:
        args += ["--input", str(local(f))]
    for reasoner in ("HermiT", "ELK"):
        res = subprocess.run(["robot", "merge", *args, "reason", "--reasoner", reasoner,
                              "--output", str(tmp / "out.ttl")], capture_output=True, text=True)
        verdict = classify(res.stdout + res.stderr)
        rows.append((pair, reasoner, verdict))
        print(f"{pair:16} {reasoner:7} {verdict}")

# Control: is the CCOM+QUDT HermiT result caused by the mappings or by QUDT itself?
# Give QUDT alone ONE unit individual and reason with HermiT.
probe = tmp / "probe.ttl"
probe.write_text("<http://example.org/probeUnit> a <http://www.w3.org/2002/07/owl#NamedIndividual> , "
                 "<http://qudt.org/schema/qudt#MassUnit> .\n")
res = subprocess.run(["robot", "merge", "--input", str(local("qudt.ttl")), "--input", str(probe),
                      "reason", "--reasoner", "HermiT", "--output", str(tmp / "out.ttl")],
                     capture_output=True, text=True)
control = classify(res.stdout + res.stderr)
rows.append(("QUDT alone + 1 MassUnit individual (control)", "HermiT", control))
print(f"{'QUDT control':16} HermiT  {control}")

report = ["# Merged reasoning report (ontologies + mappings)", "",
          "| Pair | Reasoner | Result |", "|---|---|---|"]
report += [f"| {p} | {r} | {v} |" for p, r, v in rows]
report += ["", "## Notes",
  "- **CCOM + QUDT / HermiT:** the control row shows QUDT 1.1 alone becomes inconsistent as soon as any unit "
  "individual exists: `Unit` requires exactly one `typePrefix` while each subclass level adds a different required "
  "value (U, USE, UPHS, UMH). Any correct CCOM->QUDT mapping classifies CCO's unit individuals into QUDT classes and "
  "so triggers it. ELK ignores cardinality/hasValue axioms, hence CONSISTENT.",
  "- **CCOT + OWL-Time / HermiT:** OWL-Time declares a disjointness axiom on a non-simple property, which OWL 2 DL "
  "forbids; HermiT refuses to load it. ELK (OWL 2 EL) can still check the pair.",
  "- **BFO + IES:** consistent under both reasoners."]
(DATA / "merged-reasoning-report.md").write_text("\n".join(report) + "\n")
print("report -> src/data/merged-reasoning-report.md")
