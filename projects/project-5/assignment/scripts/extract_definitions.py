#Project 5 - read the chosen CCO module and extract the IRI, Label and Defintion to a csv file

#Pulled from augment w definitions from project 3
#pulled from normalize readings for referening files 

from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS,SKOS 


# 1. Where to read and write
BASESOURCE = Path(__file__).resolve().parents[1]
IN_PATH = BASESOURCE / "src" / "cco-module.ttl"
OUT_PATH = BASESOURCE / "data" / "definitions.csv"

# 2. Load CCO ICE file
g = Graph()
g.parse(IN_PATH, format="turtle")

# 3. One row per term.  Show IRI, Label and Definition. Ignore deprecated entries
rows = []
for term, definition in g.subject_objects(SKOS.definition):
    if g.value(term, OWL.deprecated) == Literal(True):   # skip deprecated terms
        continue
    rows.append({
        "IRI": str(term),
        "label": str(g.value(term, RDFS.label)),
        "definition": str(definition),
    })

# 4. Write the file
df = pd.DataFrame(rows, columns=["IRI", "label", "definition"])
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"Wrote {OUT_PATH.name}: {len(df)} rows from {len(g)} triples")
