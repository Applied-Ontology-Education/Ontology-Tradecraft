
from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import XSD


# Locate the project directories
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_DIR / "src" / "data" / "readings_normalized.csv"
ONTOLOGY_FILE = PROJECT_DIR / "src" / "cco_merged.ttl"
OUTPUT_FILE = PROJECT_DIR / "src" / "measure_cco.ttl"


# Ontology namespaces
BFO = Namespace("http://purl.obolibrary.org/obo/")
CCO = Namespace("https://www.commoncoreontologies.org/")

# Classes used by the measurement pattern
ARTIFACT = CCO["ont00000995"]  # Material Artifact
SDC = BFO["BFO_0000020"]       # Specifically Dependent Continuant
MICE = CCO["ont00001163"]      # Measurement Information Content Entity
MU = CCO["ont00000120"]        # Measurement Unit

# Specific measurement unit classes found in the merged CCO ontology
UNIT_CLASSES = {
    "C": CCO["ont00001606"],    # Degree Celsius Measurement Unit
    "F": CCO["ont00001724"],    # Degree Fahrenheit Measurement Unit
    "pounds_per_square_inch": CCO["ont00001694"],  # Pounds Per Square Inch Measurement Unit
    "V": CCO["ont00001450"],    # Volt Measurement Unit
    "kpa_unit": MU,                  # Generic Measurement Unit for now
    "ohm": MU,                  # Generic Measurement Unit for now
}

# Properties used by the pattern
BEARER_OF = BFO["BFO_0000196"]
IS_MEASUREMENT_OF = CCO["ont00001966"]
USES_MEASUREMENT_UNIT = CCO["ont00001863"]
HAS_DECIMAL_VALUE = CCO["ont00001769"]


# Read the normalized measurements
df = pd.read_csv(DATA_FILE)

# Create the RDF graph
g = Graph()

# Bind readable prefixes for the output Turtle file
g.bind("bfo", BFO)
g.bind("cco", CCO)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)
g.bind("xsd", XSD)

print(f"Loaded {len(df)} measurement rows.")
print("Columns:", list(df.columns))


# Namespace for individuals created by this project
EX = Namespace("https://example.org/measurement/")


def make_id(value):
    """Create a simple, stable identifier from a value."""
    return (
        str(value).strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("#", "_")
    )


# Reuse the same artifact, quality, and unit individuals where appropriate
artifacts = {}
qualities = {}
units = {}

for index, row in df.iterrows():
    artifact_id = str(row["artifact_id"])
    kind = str(row["sdc_kind"])
    unit_label = str(row["unit_label"])
    timestamp = str(row["timestamp"])

    # 1. Create or reuse the artifact individual
    artifact = artifacts.setdefault(
        artifact_id,
        EX[f"artifact/{make_id(artifact_id)}"]
    )
    g.add((artifact, RDF.type, ARTIFACT))
    g.add((artifact, RDFS.label, Literal(artifact_id)))

    # 2. Create or reuse the quality (SDC) individual
    quality_key = (artifact_id, kind)
    quality = qualities.setdefault(
        quality_key,
        EX[f"quality/{make_id(artifact_id)}_{make_id(kind)}"]
    )
    g.add((quality, RDF.type, SDC))
    g.add((quality, RDF.type, BFO["BFO_0000019"]))  # BFO Quality

    # The artifact bears this quality
    g.add((artifact, BEARER_OF, quality))

    # 3. Create or reuse the measurement unit individual
    unit = units.setdefault(
        unit_label,
        EX[f"unit/{make_id(unit_label)}"]
    )
    g.add((unit, RDF.type, MU))
    g.add((unit, RDF.type, UNIT_CLASSES[unit_label]))
    g.add((unit, RDFS.label, Literal(unit_label)))

    # 4. Create a distinct measurement information entity for this row
    mice = EX[f"measurement/reading_{index + 1}"]
    g.add((mice, RDF.type, MICE))
    g.add((
        mice,
        RDFS.label,
        Literal(f"{kind} reading for {artifact_id} at {timestamp}")
    ))

    # Link the measurement to the quality and its unit
    g.add((mice, IS_MEASUREMENT_OF, quality))
    g.add((mice, USES_MEASUREMENT_UNIT, unit))

    # Store the numeric measurement value
    g.add((
        mice,
        HAS_DECIMAL_VALUE,
        Literal(str(row["value"]), datatype=XSD.decimal)
    ))

print(f"Created RDF graph with {len(g)} triples.")


# Save the RDF graph as Turtle
g.serialize(destination=str(OUTPUT_FILE), format="turtle")

print(f"RDF output saved to: {OUTPUT_FILE}")
