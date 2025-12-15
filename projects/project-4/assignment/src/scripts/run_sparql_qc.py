#!/usr/bin/env python3
from rdflib import Graph, Namespace
from pathlib import Path
import sys

# This file lives at:  .../assignment/src/scripts/run_sparql_qc.py
# The TTL file lives at: .../assignment/src/measure_cco.ttl
# The queries live at:   .../assignment/src/sparql/*.rq

BASE_DIR = Path(__file__).resolve().parents[2]   # go up from /src/scripts → /src → /assignment
DATA     = BASE_DIR / "src" / "measure_cco.ttl"
QUERIES  = BASE_DIR / "src" / "sparql"

SHOW_LIMIT = 10

def run_query(g: Graph, qpath: Path) -> int:
    q = qpath.read_text(encoding="utf-8")
    rows = list(g.query(q))
    if rows:
        print(f"❌ {qpath.name}: {len(rows)} row(s)")
        for r in rows[:SHOW_LIMIT]:
            print("   ", " | ".join(map(str, r)))
        return 1
    print(f"✅ {qpath.name}: 0 rows")
    return 0

def main() -> int:
    print("[qc] DATA   :", DATA)
    print("[qc] exists?:", DATA.exists())
    print("[qc] QUERIES:", QUERIES)
    print("[qc] exists?:", QUERIES.exists())

    if not DATA.exists():
        print(f"ERROR: data file not found: {DATA}", file=sys.stderr)
        return 2
    if not QUERIES.exists():
        print(f"ERROR: queries directory not found: {QUERIES}", file=sys.stderr)
        return 2

    g = Graph()
    
    # Bind namespaces BEFORE parsing to ensure they're preserved
    BFO = Namespace("http://purl.obolibrary.org/obo/bfo.owl#")
    CCO = Namespace("http://www.ontologyrepository.com/CommonCoreOntologies/")
    EX = Namespace("http://example.org/")
    
    g.bind("bfo", BFO)
    g.bind("cco", CCO)
    g.bind("ex", EX)
    
    g.parse(DATA, format="turtle")
    print(f"[qc] triples loaded: {len(g)}")
    
    # Verify BFO namespace is present
    bfo_triples = sum(1 for s, p, o in g if str(p).startswith("http://purl.obolibrary.org/obo/bfo.owl#"))
    print(f"[qc] BFO triples found: {bfo_triples}")

    rq_files = sorted(QUERIES.glob("*.rq"))
    print(f"[qc] found {len(rq_files)} query file(s):")
    for p in rq_files:
        print("   -", p.name)
    print("-" * 60)

    failures = 0
    for qpath in rq_files:
        failures += run_query(g, qpath)

    print("-" * 60)
    if failures:
        print(f"QC summary: ❌ {failures} failing quer{'y' if failures == 1 else 'ies'}.")
        return 1
    print("QC summary: ✅ all checks passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())