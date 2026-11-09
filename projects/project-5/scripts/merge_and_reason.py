"""Step 7: check module + accepted axioms with ELK, then save src/event-augmented.ttl
(the original module plus the accepted axioms, nothing else)."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rdflib import RDF, URIRef

from filter_candidates_hybrid import merged_graph
from generate_candidates_llm import PROJECT_DIR, asserted_axioms, load_config, load_graph


def elk_check(graph):
    """ROBOT reason fails if the ontology is inconsistent or has unsatisfiable classes."""
    robot = shutil.which("robot")
    if robot is None:
        sys.exit("robot not found on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "check.ttl"
        graph.serialize(src, format="turtle")
        r = subprocess.run([robot, "reason", "--reasoner", "ELK", "--input", str(src),
                            "--output", str(Path(tmp) / "out.ttl")],
                           capture_output=True, text=True)
    return r.returncode == 0, r.stdout + r.stderr


def main():
    cfg = load_config()
    paths = cfg["paths"]
    module = load_graph(PROJECT_DIR / paths["module"])
    bfo = load_graph(PROJECT_DIR / paths["bfo"])
    accepted = load_graph(PROJECT_DIR / paths["accepted"])

    # 1. Reason over module + BFO + accepted axioms (imports stripped so it runs offline).
    check = merged_graph(module, bfo)
    for t in accepted:
        check.add(t)
    consistent, log = elk_check(check)
    if not consistent:
        print(log)
        sys.exit("ELK: inconsistent or unsatisfiable classes (see ROBOT output above)")

    # 2. Save the original module plus the accepted axioms. Declarations of named
    #    classes/properties in accepted_el.ttl are skipped: they already exist.
    augmented = load_graph(PROJECT_DIR / paths["module"])
    for s, p, o in accepted:
        if p == RDF.type and isinstance(s, URIRef):
            continue
        augmented.add((s, p, o))
    out = PROJECT_DIR / paths["augmented"]
    augmented.serialize(out, format="turtle")

    # 3. Verify every accepted axiom made it into the saved file.
    subs_a, somes_a = asserted_axioms(accepted)
    subs_o, somes_o = asserted_axioms(load_graph(out))
    missing = (subs_a - subs_o) | (somes_a - somes_o)
    if missing:
        sys.exit(f"{len(missing)} accepted axioms are missing from {paths['augmented']}")
    n_added = len(subs_a) + len(somes_a)

    # 4. Record the result in mowl_metrics.json.
    metrics_path = PROJECT_DIR / paths["mowl_metrics"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["consistent"] = True
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"ELK: consistent, no unsatisfiable classes")
    print(f"Added {n_added} axioms -> {paths['augmented']} (all verified present)")


if __name__ == "__main__":
    main()
