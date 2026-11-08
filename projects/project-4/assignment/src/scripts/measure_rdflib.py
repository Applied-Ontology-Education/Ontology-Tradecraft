#!/usr/bin/env python3
"""
measure_rdflib.py
Reads src/data/readings_normalized.csv and emits src/measure_cco.ttl,
instantiating the CCO Measurement Design Pattern for each row.
"""
# Reminder: https://github.com/RDFLib/rdflib/blob/main/examples/simple_example.py

import os.path
from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent         # .../src/scripts
SRC_DIR    = SCRIPT_DIR.parent                        # .../src
CSV_IN     = SRC_DIR / "data" / "readings_normalized.csv"
TTL_OUT    = SRC_DIR / "measure_cco.ttl"

# Namespaces
EX  = Namespace("https://example.org/project4/") ## Example
CCO = Namespace("https://www.commoncoreontologies.org/")
OBO = Namespace("http://purl.obolibrary.org/obo/")

# IRIs See cco_merged.ttl

# Classes
ARTIFACT = CCO["ont00000995"]      # Material Artifact
SDC      = OBO["BFO_0000020"]      # Specifically Dependent Continuant
MICE     = CCO["ont00001163"]      # Measurement Information Content Entity
MU       = CCO["ont00000120"]      # Measurement Unit

# Object properties
BEARER_OF         = OBO["BFO_0000196"]   # bearer of
IS_MEASUREMENT_OF = CCO["ont00001966"]   # is a measurement of
USES_MU           = CCO["ont00001863"]   # uses measurement unit

# Data property
HAS_DECIMAL_VALUE = CCO["ont00001769"]   # has decimal value

# clean spaces
def _safe(s: str) -> str:
    """Replace characters that aren't valid in IRIs."""
    return str(s).replace(" ", "_").replace(":", "-")

# Build IRIs
def artifact_iri(artifact_id: str) -> URIRef:
    return EX[f"artifact_{_safe(artifact_id)}"]

def sdc_iri(artifact_id: str, sdc_kind: str) -> URIRef:
    return EX[f"sdc_{_safe(artifact_id)}_{_safe(sdc_kind)}"]

def unit_iri(unit_label: str) -> URIRef:
    return EX[f"unit_{_safe(unit_label)}"]

def mice_iri(artifact_id: str, sdc_kind: str, timestamp: str) -> URIRef:
    return EX[f"mice_{_safe(artifact_id)}_{_safe(sdc_kind)}_{_safe(timestamp)}"]

if __name__ == "__main__":
    if not CSV_IN.exists():
        raise FileNotFoundError(
            f"Missing {CSV_IN}. Run normalize_readings.py first."
        )
    df = pd.read_csv(CSV_IN)
    print(f"[measure_rdflib] read {len(df)} rows from {CSV_IN}")

    store = Graph()
    store.bind("", EX)
    store.bind("cco", CCO)
    store.bind("obo", OBO)

    # Build the four-instance pattern for each row.
    # https://rdflib.readthedocs.io/en/stable/intro_to_creating_rdf/
    for _, row in df.iterrows():
        art  = artifact_iri(row["artifact_id"])
        sdc  = sdc_iri(row["artifact_id"], row["sdc_kind"])
        unit = unit_iri(row["unit_label"])
        mice = mice_iri(row["artifact_id"], row["sdc_kind"], row["timestamp"])

        # triples
        store.add((art,  RDF.type, ARTIFACT))
        store.add((sdc,  RDF.type, SDC))
        store.add((mice, RDF.type, MICE))
        store.add((unit, RDF.type, MU))

        # relationships
        store.add((art,  BEARER_OF,         sdc))
        store.add((mice, IS_MEASUREMENT_OF, sdc))
        store.add((mice, USES_MU,           unit))
        store.add((mice, HAS_DECIMAL_VALUE,
                   Literal(row["value"], datatype=XSD.decimal)))

        # labels
        store.add((art,  RDFS.label, Literal(str(row["artifact_id"]))))
        store.add((sdc,  RDFS.label, Literal(str(row["sdc_kind"]))))
        store.add((unit, RDFS.label, Literal(str(row["unit_label"]))))

    TTL_OUT.parent.mkdir(parents=True, exist_ok=True)
    store.serialize(destination=str(TTL_OUT), format="turtle")
    print(f"[measure_rdflib] wrote {len(store)} triples to {TTL_OUT}")
