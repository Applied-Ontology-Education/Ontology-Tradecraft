import csv
from pathlib import Path

import rdflib

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_DIR / "src" / "EventOntology.ttl"
OUTPUT_PATH = PROJECT_DIR / "data" / "definitions.csv"

g = rdflib.Graph()
g.parse(INPUT_PATH, format="turtle")

query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?IRI ?label ?definition
WHERE {
    ?IRI rdfs:label ?label .
    OPTIONAL { ?IRI skos:definition ?definition . }
}
"""

results = g.query(query)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["IRI", "label", "definition"])
    for row in results:
        writer.writerow([str(row.IRI), str(row.label), str(row.definition) if row.definition else ""])