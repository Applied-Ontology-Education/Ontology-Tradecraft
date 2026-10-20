##Build src/measure_cco.ttl from src/data/readings_normalized.csv using the measurement design pattern:
##
##  artifact --bearer of--> SDC
##  MICE --is measurement of--> SDC
##  MICE --uses measurement unit--> unit
##  MICE --has decimal value--> number

from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS,SKOS, XSD

# 1. Where to read and write
SRC = Path(__file__).resolve().parents[1]
CSV_PATH = SRC / "data" / "readings_normalized.csv"
OUT_PATH = SRC / "measure_cco.ttl"

# 2. Namespaces
EX = Namespace("http://example.org/project4/") #my ontology namespace
CCO = Namespace("https://www.commoncoreontologies.org/")
OBO = Namespace("http://purl.obolibrary.org/obo/")

# 3. The IRIs from the design pattern  
ARTIFACT = CCO.ont00000995            # Material Artifact
SDC = OBO.BFO_0000020                 # Specifically Dependent Continuant
MICE = CCO.ont00001163                # Measurement Information Content Entity
UNIT = CCO.ont00000120                # Measurement Unit

BEARER_OF = OBO.BFO_0000196
IS_MEASUREMENT_OF = CCO.ont00001966
USES_UNIT = CCO.ont00001863
HAS_DECIMAL_VALUE = CCO.ont00001769
HAS_DATETIME_VALUE = CCO.ont00001767

# 4. unit_label in the CSV mapping to the measurement unit IRI
UNIT_IRI = {
    "F": CCO.ont00001724,
    "C": CCO.ont00001606,
    "[psi]": CCO.ont00001694,
    "Pa": CCO.ont00001559,
    "V": CCO.ont00001450,
    "ohm": EX.unit_ohm,   # CCO has no ohm, 
}

# 4b. Our own classes (copied from Project4 design.ttl)
#     class: (label, definition, parent class)
CLASSES = {
    EX.Device: ("Device",
                "A Material Artifact that is designed to perform a specific operational function within an engineered system.",
                ARTIFACT),
    EX.Boiler: ("Boiler",
                "A Device that is designed to heat water or generate steam by transferring heat from a fuel or energy source.",
                EX.Device),
    EX.Chiller: ("Chiller",
                 "A Device that is designed to remove heat from a liquid by refrigeration in order to supply chilled liquid for cooling.",
                 EX.Device),
    EX.Electrical_Circuit: ("Electrical Circuit",
                            "A Device that consists of interconnected electrical components forming a closed conductive path through which electric current flows.",
                            EX.Device),
}

# artifact_id in the CSV -> its most specific class
DEVICE_CLASS = {
    "Boiler-07": EX.Boiler,
    "Chiller-3": EX.Chiller,
    "Chiller-4": EX.Chiller,
    "Circuit-12": EX.Electrical_Circuit,
    "Pump-A1": CCO.ont00000116,   # Pump (a CCO class)
}


# 5. Start the graph
g = Graph()
g.bind("ex", EX)
g.bind("cco", CCO)
g.bind("obo", OBO)
g.bind("skos", SKOS)

ontology = EX.measurements
g.add((ontology, RDF.type, OWL.Ontology))
g.add((ontology, RDFS.label, Literal("Project 4 Measurement Graph", lang="en")))

# declare our classes: type, parent, label, definition
for cls, (label, definition, parent) in CLASSES.items():
    g.add((cls, RDF.type, OWL.Class))
    g.add((cls, RDFS.subClassOf, parent))
    g.add((cls, RDFS.label, Literal(label, lang="en")))
    g.add((cls, SKOS.definition, Literal(definition, lang="en")))

# copied from CCO: Material Artifact is a subclass of BFO material entity.
# Lets query 7 trace Device up to BFO without loading all of CCO.
g.add((ARTIFACT, RDFS.subClassOf, OBO.BFO_0000040))

# 6. One pass per CSV row #use Boiler 7 as an example of a device, temperature as an example of a kind, F as an example of a unit_label, 212.0 as an example of a value, and 2024-03-17T23:03:00Z as an example of a timestamp
df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)

for i, row in df.iterrows():
    device = row["artifact_id"]      # e.g. Boiler-07
    kind = row["sdc_kind"]           # e.g. temperature
    unit_label = row["unit_label"]   # e.g. F
    value = row["value"]             # e.g. 212.0
    time = row["timestamp"]          # e.g. 2024-03-17T23:03:00Z

    artifact = EX[device]
    sdc = EX[f"sdc_{device}_{kind}"]
    mice = EX[f"mice_{i + 1:04d}"]
    unit = UNIT_IRI[unit_label]      # stops with KeyError if a unit is not in the dictionary

    # the artifact bears the quality
    g.add((artifact, RDF.type, ARTIFACT))
    g.add((artifact, RDF.type, EX.Device))
    g.add((artifact, RDF.type, DEVICE_CLASS.get(device, EX.Device)))
    g.add((artifact, RDFS.label, Literal(device, lang="en")))
    g.add((artifact, BEARER_OF, sdc))

    # the quality
    g.add((sdc, RDF.type, SDC))
    g.add((sdc, RDFS.label, Literal(f"{kind} of {device}", lang="en")))

    # the unit
    g.add((unit, RDF.type, UNIT))
    g.add((unit, RDFS.label, Literal(unit_label, lang="en")))

    # the measurement
    g.add((mice, RDF.type, MICE))
    g.add((mice, IS_MEASUREMENT_OF, sdc))
    g.add((mice, USES_UNIT, unit))
    g.add((mice, HAS_DECIMAL_VALUE, Literal(value, datatype=XSD.decimal)))
    g.add((mice, HAS_DATETIME_VALUE, Literal(time, datatype=XSD.dateTime)))
    g.add((mice, RDFS.label, Literal(f"{device} {kind} {value} {unit_label} at {time}", lang="en")))

# 7. Write the file
g.serialize(OUT_PATH, format="turtle")
print(f"Wrote {OUT_PATH.name}: {len(df)} rows, {len(g)} triples")