"""Step 2: normalize and enrich definitions with an LLM -> data/definitions_enriched.csv.

Each definition is rewritten into CCO's canonical genus-differentia form ("A <genus> that <differentia>"),
with abbreviations expanded and ambiguity removed. The genus is anchored to the asserted superclass so
terminology stays aligned with CCO."""
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, IMPORTS, MODULE, ask_json, save_cache  # noqa: E402

SYSTEM = (
    "You are an ontology editor for the Common Core Ontologies (CCO), a BFO-based suite. "
    "You rewrite textual definitions into CCO's canonical Aristotelian form and reply with JSON only."
)
PROMPT = """Normalize this CCO class definition.

Class label: {label}
Asserted superclass label(s): {parents}
Original definition: {definition}

Rules:
- Canonical form: "A(n) <genus> that <differentia>." where <genus> is the asserted superclass label.
- Expand every abbreviation or acronym; remove ambiguity; keep CCO terminology (e.g. "designed to", "bearer of").
- Do not add facts that the original definition does not state or clearly imply.

Reply with exactly this JSON object:
{{"normalized_definition": "...", "genus": "...", "differentia": "...", "expanded_abbreviations": ["..."], "changes": "one short sentence"}}"""


def labels():
    g = Graph()
    for f in [MODULE, *IMPORTS.glob("*.ttl")]:
        g.parse(f)
    return g


def main():
    rows = list(csv.DictReader(open(DATA / "definitions.csv")))
    g = labels()

    def parents(iri):
        ps = [str(g.value(o, RDFS.label) or o) for o in g.objects(URIRef(iri), RDFS.subClassOf) if isinstance(o, URIRef)]
        return "; ".join(sorted(ps)) or "(none)"

    def work(r):
        out = ask_json(SYSTEM, PROMPT.format(label=r["label"], parents=parents(r["IRI"]), definition=r["definition"]))
        return {**r,
                "normalized_definition": " ".join(str(out.get("normalized_definition", r["definition"])).split()),
                "genus": out.get("genus", ""), "differentia": out.get("differentia", ""),
                "expanded_abbreviations": "; ".join(out.get("expanded_abbreviations") or []),
                "changes": out.get("changes", "")}

    with ThreadPoolExecutor(8) as ex:
        enriched = list(ex.map(work, rows))
    save_cache()
    cols = ["IRI", "label", "definition", "normalized_definition", "genus", "differentia", "expanded_abbreviations", "changes"]
    with open(DATA / "definitions_enriched.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(enriched)
    print(f"preprocess_definitions_llm: {len(enriched)} definitions normalized -> data/definitions_enriched.csv")


if __name__ == "__main__":
    main()
