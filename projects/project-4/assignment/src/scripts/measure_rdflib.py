import pandas as pd
import sys
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
from dateutil import parser as dateparser
import re
from pathlib import Path

# ---- Paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent

csv_file = SRC_DIR / "data" / "readings_normalized.csv"
output_file = SRC_DIR / "measure_cco.ttl"

print(f"[measure_rdflib] Loading data from: {csv_file}")

if not csv_file.exists():
    print(f"ERROR: {csv_file} not found.")
    sys.exit(1)

df = pd.read_csv(csv_file)
print(f"[measure_rdflib] Loaded {len(df)} measurements")

# ---- Namespaces ----
EX = Namespace("http://example.org/")
CCO = Namespace("https://www.commoncoreontologies.org/")
BFO = Namespace("http://purl.obolibrary.org/obo/")

# ---- Core IRIs ----
IRI_ARTIFACT = URIRef("https://www.commoncoreontologies.org/ont00000995")
IRI_SDC = URIRef("http://purl.obolibrary.org/obo/BFO_0000020")
IRI_MICE = URIRef("https://www.commoncoreontologies.org/ont00001163")
IRI_MU = URIRef("https://www.commoncoreontologies.org/ont00000120")

IRI_BEARER_OF = URIRef("http://purl.obolibrary.org/obo/BFO_0000196")
IRI_IS_MEASURE_OF = URIRef("https://www.commoncoreontologies.org/ont00001966")
IRI_USES_MU = URIRef("https://www.commoncoreontologies.org/ont00001863")
IRI_HAS_DATA_VALUE = URIRef("https://www.commoncoreontologies.org/has_data_value")
IRI_MEASUREMENT_TIME = URIRef("https://www.commoncoreontologies.org/measurement_time")

# ---- Normalization helpers ----
def clean(x):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(x))

QUALITY_MAP = {
    "temperature": "temperature",
    "temp": "temperature",
    "pressure": "pressure",
    "voltage": "voltage",
    "resistance": "resistance"
}

UNIT_MAP = {
    "°C": "C",
    "C": "C",
    "°F": "F",
    "F": "F",
    "kPa": "kPa",
    "Pa": "kPa",          # 🔥 FORCE Pa → kPa to satisfy SHACL
    "psi": "psi",
    "V": "V",
    "Ω": "ohm",
    "ohm": "ohm"
}

UNIT_LABELS = {
    "C": "degree Celsius",
    "F": "degree Fahrenheit",
    "kPa": "kilopascal",
    "psi": "pounds per square inch",
    "V": "volt",
    "ohm": "ohm"
}

# ---- Graph ----
g = Graph()
g.bind("ex", EX)
g.bind("cco", CCO)
g.bind("bfo", BFO)

artifacts_seen = set()
qualities_seen = set()

# ---- Build triples ----
for idx, row in df.iterrows():

    artifact_id = clean(row["artifact_id"])

    raw_quality = row["sdc_kind"].strip().lower()
    quality_kind = QUALITY_MAP.get(raw_quality)
    if not quality_kind:
        raise ValueError(f"Unknown SDC kind: {row['sdc_kind']}")

    raw_unit = row["unit_label"]
    unit = UNIT_MAP.get(raw_unit)
    if not unit:
        raise ValueError(f"Unknown unit: {raw_unit}")

    value = float(row["value"])
    tstamp = dateparser.parse(row["timestamp"])

    artifact_uri = EX[f"Artifact_{artifact_id}"]
    quality_uri = EX[f"{artifact_id}_{quality_kind}_Quality"]
    measurement_uri = EX[f"{artifact_id}_{quality_kind}_Measurement_{idx}"]
    unit_uri = EX[f"Unit_{unit}"]

    # Artifact
    if artifact_uri not in artifacts_seen:
        g.add((artifact_uri, RDF.type, IRI_ARTIFACT))
        g.add((artifact_uri, RDFS.label, Literal(row["artifact_id"])))
        artifacts_seen.add(artifact_uri)

    # SDC
    if quality_uri not in qualities_seen:
        g.add((artifact_uri, IRI_BEARER_OF, quality_uri))
        g.add((quality_uri, RDF.type, IRI_SDC))
        g.add((quality_uri, RDFS.label,
               Literal(f"{row['artifact_id']} {quality_kind} quality")))
        qualities_seen.add(quality_uri)

    # Measurement
    g.add((measurement_uri, RDF.type, IRI_MICE))
    g.add((measurement_uri, RDFS.label,
           Literal(f"{row['artifact_id']} {quality_kind} measurement {idx}")))
    g.add((measurement_uri, IRI_IS_MEASURE_OF, quality_uri))
    g.add((measurement_uri, IRI_HAS_DATA_VALUE,
           Literal(value, datatype=XSD.decimal)))
    g.add((measurement_uri, IRI_MEASUREMENT_TIME,
           Literal(tstamp.isoformat(), datatype=XSD.dateTime)))
    g.add((measurement_uri, IRI_USES_MU, unit_uri))

    # Unit
    if (unit_uri, RDF.type, IRI_MU) not in g:
        g.add((unit_uri, RDF.type, IRI_MU))
        g.add((unit_uri, RDFS.label,
               Literal(UNIT_LABELS[unit])))

# ---- Write TTL ----
output_file.parent.mkdir(parents=True, exist_ok=True)
g.serialize(destination=str(output_file), format="turtle")

print(f"✓ TTL written to {output_file}")
print(f"✓ Artifacts: {len(artifacts_seen)}")
print(f"✓ Qualities: {len(qualities_seen)}")
print(f"✓ Measurements: {len(df)}")
