# src/scripts/measure_rdflib.py

from pathlib import Path

import pandas as pd
from rdflib import Graph, Namespace, RDF, RDFS, XSD, Literal, URIRef

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------

# This file lives in: .../projects/project-4/assignment/src/scripts
# assignment root is two levels up
ASSIGNMENT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ASSIGNMENT_ROOT / "src"
DATA_DIR = SRC_DIR / "data"

READINGS_CSV = DATA_DIR / "readings_normalized.csv"
CCO_TTL = ASSIGNMENT_ROOT / "../assignment/src/cco_merged.ttl"
OUT_TTL = SRC_DIR / "measure_cco.ttl"

# --------------------------------------------------------------------
# Namespaces & CCO/BFO bindings
# --------------------------------------------------------------------

BFO = Namespace("http://purl.obolibrary.org/obo/")
CCO = Namespace("https://www.commoncoreontologies.org/")
EX  = Namespace("http://example.org/ontology-tradecraft/")  # project-local namespace

# Classes (from cco_merged.ttl)
CCO_ARTIFACT_CLASS = URIRef("https://www.commoncoreontologies.org/ont00000995")  # Material Artifact
BFO_SDC_CLASS      = URIRef("http://purl.obolibrary.org/obo/BFO_0000020")        # specifically dependent continuant
CCO_MICE_CLASS     = URIRef("https://www.commoncoreontologies.org/ont00001163")  # Measurement Information Content Entity
CCO_MU_CLASS       = URIRef("https://www.commoncoreontologies.org/ont00000120")  # Measurement Unit

# Object properties
BFO_BEARER_OF             = URIRef("http://purl.obolibrary.org/obo/BFO_0000196")   # bearer of
CCO_IS_MEASUREMENT_OF     = URIRef("https://www.commoncoreontologies.org/ont00001966")  # is a measurement of
CCO_USES_MEASUREMENT_UNIT = URIRef("https://www.commoncoreontologies.org/ont00001863")  # uses measurement unit

# Data properties
CCO_HAS_DOUBLE_VALUE   = URIRef("https://www.commoncoreontologies.org/ont00001770")  # has double value
CCO_HAS_DATETIME_VALUE = URIRef("https://www.commoncoreontologies.org/ont00001767")  # has datetime value


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def slugify(s: str) -> str:
    """
    Turn an arbitrary string into a safe-ish local name fragment.
    """
    return (
        str(s)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def build_graph() -> Graph:
    """
    Build an RDF graph that instantiates the measurement design pattern for
    each row in src/data/readings_normalized.csv.
    """
    # 1) Start from merged CCO/BFO graph
    g = Graph()
    g.parse(CCO_TTL, format="turtle")

    # 2) Bind prefixes for nice Turtle output
    g.bind("bfo", BFO)
    g.bind("cco", CCO)
    g.bind("ex", EX)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # 3) Load normalized readings
    df = pd.read_csv(READINGS_CSV)

    required_cols = ["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"readings_normalized.csv is missing required columns: {missing}")

    # Caches so we don't mint duplicate individuals for the same artifact / SDC / unit
    artifact_cache = {}
    sdc_cache = {}
    unit_cache = {}

    # 4) For each row, instantiate the pattern:
    #    Artifact (art-inst)
    #    SDC      (sdc-inst) bearer_of Artifact
    #    MICE     (mice-inst) is_measurement_of SDC; has double value; uses measurement unit MU
    #    MU       (unit-inst)
    for idx, row in df.iterrows():
        artifact_id = str(row["artifact_id"])
        sdc_kind    = str(row["sdc_kind"])
        unit_label  = str(row["unit_label"])
        value       = row["value"]
        timestamp   = str(row["timestamp"])

        # -----------------------------
        # Artifact individual (art-inst)
        # -----------------------------
        if artifact_id in artifact_cache:
            art_iri = artifact_cache[artifact_id]
        else:
            art_iri = EX[f"artifact/{slugify(artifact_id)}"]
            artifact_cache[artifact_id] = art_iri

            g.add((art_iri, RDF.type, CCO_ARTIFACT_CLASS))
            g.add((art_iri, RDFS.label, Literal(artifact_id)))

        # -------------------------------------------------------
        # Specifically Dependent Continuant (sdc-inst)
        # model this as "the {sdc_kind} of {artifact_id}"
        # -------------------------------------------------------
        sdc_key = (artifact_id, sdc_kind)
        if sdc_key in sdc_cache:
            sdc_iri = sdc_cache[sdc_key]
        else:
            sdc_iri = EX[f"sdc/{slugify(artifact_id)}_{slugify(sdc_kind)}"]
            sdc_cache[sdc_key] = sdc_iri

            g.add((sdc_iri, RDF.type, BFO_SDC_CLASS))
            label = f"{sdc_kind} of {artifact_id}"
            g.add((sdc_iri, RDFS.label, Literal(label)))

            # bfo:bearer_of(sdc-inst, art-inst)
            g.add((art_iri, BFO_BEARER_OF, sdc_iri))

        # -----------------------------
        # Measurement Unit (unit-inst)
        # -----------------------------
        unit_key = unit_label
        if unit_key in unit_cache:
            unit_iri = unit_cache[unit_key]
        else:
            unit_iri = EX[f"unit/{slugify(unit_label)}"]
            unit_cache[unit_key] = unit_iri

            g.add((unit_iri, RDF.type, CCO_MU_CLASS))
            g.add((unit_iri, RDFS.label, Literal(unit_label)))

        # ------------------------------------------
        # Measurement Information Content Entity
        # (mice-inst)
        # ------------------------------------------
        mice_iri = EX[f"measurement/{idx}"]
        g.add((mice_iri, RDF.type, CCO_MICE_CLASS))

        # cco:is a measurement of (mice-inst, sdc-inst)
        g.add((mice_iri, CCO_IS_MEASUREMENT_OF, sdc_iri))

        # cco:uses measurement unit (mice-inst, unit-inst)
        g.add((mice_iri, CCO_USES_MEASUREMENT_UNIT, unit_iri))

        # Numeric value as xsd:double via cco:has double value
        try:
            numeric_value = float(value)
            g.add(
                (
                    mice_iri,
                    CCO_HAS_DOUBLE_VALUE,
                    Literal(numeric_value, datatype=XSD.double),
                )
            )
        except Exception:
            # Fallback: store as plain literal if conversion fails
            g.add(
                (
                    mice_iri,
                    CCO_HAS_DOUBLE_VALUE,
                    Literal(str(value)),
                )
            )

        # Timestamp as xsd:dateTime via cco:has datetime value
        if timestamp:
            g.add(
                (
                    mice_iri,
                    CCO_HAS_DATETIME_VALUE,
                    Literal(timestamp, datatype=XSD.dateTime),
                )
            )

    return g


def main():
    g = build_graph()
    OUT_TTL.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(OUT_TTL, format="turtle")
    print(f"Wrote {OUT_TTL.relative_to(ASSIGNMENT_ROOT)} with {len(g)} triples.")


if __name__ == "__main__":
    main()
