"""Step 1: extract skos:definition text for every class declared in the module -> data/definitions.csv."""
import csv
import sys

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import DATA, MODULE, MODULE_ALIAS, SKOS_DEF  # noqa: E402


def main():
    src = MODULE_ALIAS if MODULE_ALIAS.exists() else MODULE
    if MODULE_ALIAS.exists() and MODULE_ALIAS.read_bytes() != MODULE.read_bytes():
        raise SystemExit("src/module.ttl differs from src/cco-module.ttl; they must be identical copies.")
    g = Graph().parse(src)
    rows = []
    for c in set(g.subjects(RDF.type, OWL.Class)):
        if not isinstance(c, URIRef):
            continue
        d = g.value(c, URIRef(SKOS_DEF))
        if not isinstance(d, Literal):
            continue
        rows.append({"IRI": str(c), "label": str(g.value(c, RDFS.label) or ""), "definition": " ".join(str(d).split())})
    rows.sort(key=lambda r: r["label"].lower())
    DATA.mkdir(exist_ok=True)
    with open(DATA / "definitions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["IRI", "label", "definition"])
        w.writeheader()
        w.writerows(rows)
    print(f"extract_definitions: {len(rows)} definitions from {src.name} -> data/definitions.csv")


if __name__ == "__main__":
    main()
