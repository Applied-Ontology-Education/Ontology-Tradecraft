from pathlib import Path

import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, XSD, OWL

# This file lives at:   .../assignment/src/scripts/measure_rdflib.py
# The CSV lives at:     .../assignment/src/data/readings_normalized.csv
# The output goes to:   .../assignment/src/measure_cco.ttl

SRC_DIR = Path(__file__).resolve().parents[1]   # .../assignment/src
CSV_PATH = SRC_DIR / "data" / "readings_normalized.csv"
OUT_PATH = SRC_DIR / "measure_cco.ttl"

BFO = Namespace("http://purl.obolibrary.org/obo/")
CCO = Namespace("https://www.commoncoreontologies.org/")
DATA = Namespace("http://example.org/project4/data#")

df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
print(f"[measure] loaded {len(df)} rows from {CSV_PATH}")
import re


def _slug(s: str) -> str:
    """Lowercase, replace anything that isn't alphanumeric/hyphen with a hyphen,
    and collapse repeats, so the string is safe as a URI local name."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


artifact_ids = sorted(df["artifact_id"].unique())
sdc_pairs = sorted(set(zip(df["artifact_id"], df["sdc_kind"])))
unit_labels = sorted(df["unit_label"].unique())

artifact_iris = {aid: DATA[f"artifact-{_slug(aid)}"] for aid in artifact_ids}

sdc_iris = {
    (aid, kind): DATA[f"sdc-{_slug(aid)}-{_slug(kind)}"]
    for aid, kind in sdc_pairs
}

mu_iris = {unit: DATA[f"mu-{_slug(unit)}"] for unit in unit_labels}

print(f"[measure] {len(artifact_iris)} artifacts, {len(sdc_iris)} SDCs, {len(mu_iris)} measurement units")
# Exact IRIs, confirmed against cco_merged.ttl
ARTIFACT_TYPE = CCO["ont00000995"]   # Material Artifact
SDC_TYPE      = BFO["BFO_0000020"]   # specifically dependent continuant (required by CI)
QUALITY_TYPE  = BFO["BFO_0000019"]   # quality — genuinely true of all four sdc_kinds here
MU_TYPE       = CCO["ont00000120"]   # Measurement Unit
BEARER_OF     = BFO["BFO_0000196"]

# More specific CCO subclass per sdc_kind, where one genuinely exists in cco_merged.ttl.
# voltage/resistance have no CCO quality subclass (only unrelated "Artifact Function"
# classes), so they stay at Quality + SDC rather than being assigned something that
# isn't actually there.
SPECIFIC_SDC_TYPE = {
    "temperature": CCO["ont00000441"],  # Temperature
    "pressure": CCO["ont00000380"],     # Pressure
}

# More specific CCO Measurement Unit subclass per unit_label, where one exists.
# "ohm" has no CCO subclass, stays at plain Measurement Unit.
SPECIFIC_MU_TYPE = {
    "Pa": CCO["ont00001559"],   # Pascal Measurement Unit
    "F": CCO["ont00001724"],    # Degree Fahrenheit Measurement Unit
    "C": CCO["ont00001606"],    # Degree Celsius Measurement Unit
    "V": CCO["ont00001450"],    # Volt Measurement Unit
}

g = Graph()
g.bind("bfo", BFO)
g.bind("cco", CCO)
g.bind("data", DATA)

# Artifacts
for aid, iri in artifact_iris.items():
    g.add((iri, RDF.type, ARTIFACT_TYPE))
    g.add((iri, RDFS.label, Literal(aid)))

# SDCs, typed + linked back to their bearing Artifact
for (aid, kind), iri in sdc_iris.items():
    g.add((iri, RDF.type, SDC_TYPE))
    g.add((iri, RDF.type, QUALITY_TYPE))
    specific = SPECIFIC_SDC_TYPE.get(kind)
    if specific is not None:
        g.add((iri, RDF.type, specific))
    g.add((artifact_iris[aid], BEARER_OF, iri))

# Measurement Units
for unit, iri in mu_iris.items():
    g.add((iri, RDF.type, MU_TYPE))
    specific = SPECIFIC_MU_TYPE.get(unit)
    if specific is not None:
        g.add((iri, RDF.type, specific))

print(f"[measure] graph has {len(g)} triples after Artifacts/SDCs/MUs")
# Exact property/class IRIs, confirmed against cco_merged.ttl
MICE_TYPE          = CCO["ont00001163"]  # Measurement Information Content Entity
IS_MEASURE_OF      = CCO["ont00001966"]
USES_MEASUREMENT_U = CCO["ont00001863"]
HAS_DECIMAL_VALUE  = CCO["ont00001769"]

# One MICE per CSV row — this is the only loop that runs per-row rather than per-unique-key
for i, row in df.iterrows():
    mice_iri = DATA[f"mice-{i:04d}"]
    sdc_iri = sdc_iris[(row["artifact_id"], row["sdc_kind"])]
    mu_iri = mu_iris[row["unit_label"]]

    g.add((mice_iri, RDF.type, MICE_TYPE))
    g.add((mice_iri, IS_MEASURE_OF, sdc_iri))
    g.add((mice_iri, USES_MEASUREMENT_U, mu_iri))
    g.add((mice_iri, HAS_DECIMAL_VALUE, Literal(float(row["value"]), datatype=XSD.decimal)))

print(f"[measure] graph has {len(g)} triples after MICE")

# Ontology header — makes this file self-describing like any real OWL/TTL file,
# and gives the "ontology must have a title" QC query something real to check.
ONTOLOGY_IRI = URIRef("http://example.org/project4/data")
g.add((ONTOLOGY_IRI, RDF.type, OWL.Ontology))
g.add((ONTOLOGY_IRI, RDFS.label, Literal("CCO Measurement Instance Data", lang="en")))

g.serialize(destination=str(OUT_PATH), format="turtle")
print(f"[measure] wrote {OUT_PATH}")