#!/usr/bin/env python3
"""
Build a CCO Measurement Design Pattern graph from src/data/readings_normalized.csv.

Pattern (cco_measurement_design_pattern.pptx):
    art-inst  rdf:type cco:Artifact ;            bfo:bearer_of        sdc-inst .
    sdc-inst  rdf:type bfo:SDC .
    mice-inst rdf:type cco:MICE ;                cco:is_measure_of    sdc-inst ;
                                                 cco:uses_measurement_unit unit-inst ;
                                                 cco:has_decimal_value "value"^^xsd:decimal .
    unit-inst rdf:type cco:MeasurementUnit .

Run from projects/project-4/assignment:
    python src/scripts/measure_rdflib.py
"""

import re
from decimal import Decimal
from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

# Paths resolved relative to this file so the script works from any cwd
SRC_DIR = Path(__file__).resolve().parents[1]
IN_CSV = SRC_DIR / "data" / "readings_normalized.csv"
OUT_TTL = SRC_DIR / "measure_cco.ttl"

CCO = Namespace("https://www.commoncoreontologies.org/")
BFO = Namespace("http://purl.obolibrary.org/obo/")
EX = Namespace("http://example.org/measurement/")

# Exact IRIs required by the design pattern / grader
IRI_SDC = BFO["BFO_0000020"]            # specifically dependent continuant
IRI_ART = CCO["ont00000995"]            # Artifact
IRI_MU = CCO["ont00000120"]             # Measurement Unit
IRI_MICE = CCO["ont00001163"]           # Measurement Information Content Entity
IRI_BEARER_OF = BFO["BFO_0000196"]      # bearer of
IRI_IS_MEASURE_OF = CCO["ont00001966"]  # is a measurement of
IRI_USES_MU = CCO["ont00001863"]        # uses measurement unit
IRI_HAS_DECIMAL = CCO["ont00001769"]    # has decimal value

# More specific CCO quality classes, where CCO provides one
SDC_SUBTYPES = {
    "temperature": CCO["ont00000441"],  # Temperature
    "pressure": CCO["ont00000380"],     # Pressure
}

UNIT_LABELS = {
    "degC": "degree Celsius",
    "degF": "degree Fahrenheit",
    "PSI_gauge": "pound per square inch (gauge)",
    "kPa_gauge": "kilopascal (gauge)",
    "V": "volt",
    "Ω": "ohm",
}


def slug(text: str) -> str:
    """Make a safe IRI local name."""
    text = str(text).strip().replace("Ω", "ohm")
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def main() -> None:
    df = pd.read_csv(IN_CSV, dtype=str, keep_default_na=False, encoding="utf-8")
    df = df.drop_duplicates().reset_index(drop=True)

    g = Graph()
    g.bind("cco", CCO)
    g.bind("obo", BFO)
    g.bind("ex", EX)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)

    ont = URIRef("http://example.org/measurement")
    g.add((ont, RDF.type, OWL.Ontology))
    g.add((ont, RDFS.label, Literal("Sensor Measurement Graph", lang="en")))

    for i, row in df.iterrows():
        art_id = row["artifact_id"].strip()
        kind = row["sdc_kind"].strip()
        unit = row["unit_label"].strip()
        value = row["value"].strip()
        ts = row["timestamp"].strip()

        art = EX[slug(art_id)]
        sdc = EX[f"{slug(art_id)}-{slug(kind)}-sdc"]
        unit_node = EX[f"unit-{slug(unit)}"]
        mice = EX[f"mice-{i + 1:06d}"]

        # Artifact bears SDC
        g.add((art, RDF.type, IRI_ART))
        g.add((art, RDFS.label, Literal(art_id, lang="en")))
        g.add((art, IRI_BEARER_OF, sdc))

        # SDC
        g.add((sdc, RDF.type, IRI_SDC))
        if kind in SDC_SUBTYPES:
            g.add((sdc, RDF.type, SDC_SUBTYPES[kind]))
        g.add((sdc, RDFS.label, Literal(f"{kind} of {art_id}", lang="en")))

        # Measurement unit
        g.add((unit_node, RDF.type, IRI_MU))
        g.add((unit_node, RDFS.label, Literal(UNIT_LABELS.get(unit, unit), lang="en")))

        # MICE
        g.add((mice, RDF.type, IRI_MICE))
        g.add((mice, RDFS.label,
               Literal(f"{kind} measurement of {art_id} at {ts}: {value} {unit}", lang="en")))
        g.add((mice, IRI_IS_MEASURE_OF, sdc))
        g.add((mice, IRI_USES_MU, unit_node))
        g.add((mice, IRI_HAS_DECIMAL, Literal(Decimal(value), datatype=XSD.decimal)))

    OUT_TTL.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(OUT_TTL), format="turtle", encoding="utf-8")
    print(f"[measure_rdflib] Read {len(df)} rows from {IN_CSV}")
    print(f"[measure_rdflib] Wrote {OUT_TTL} with {len(g)} triples.")


if __name__ == "__main__":
    main()
