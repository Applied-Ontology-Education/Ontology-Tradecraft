from __future__ import annotations
from pathlib import Path
import csv
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

SCRIPT_DIR = Path(__file__).resolve().parent          # .../src/scripts
SRC_DIR    = SCRIPT_DIR.parent                         # .../src
DATA_DIR   = SRC_DIR / "data"                          # .../src/data
OUT        = SRC_DIR / "measure_cco.ttl"

# 1. Initialize the Graph
g = Graph()

# 2. Define your Namespaces
EX = Namespace("http://example.org/")
CCO = Namespace("https://www.commoncoreontologies.org/")
OBO = Namespace("http://purl.obolibrary.org/obo/")
g.bind("ex", EX)
g.bind("cco", CCO)
g.bind("obo", OBO)

# 3. Read the CSV and populate the graph
with open(DATA_DIR / "readings_normalized.csv", mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    # unit_label -> existing CCO unit individual (already typed in CCO, so no rdf:type needed)
    UNITS = {
        "F":   CCO.ont00001724,   # Degree Fahrenheit Measurement Unit
        "C":   CCO.ont00001606,   # Degree Celsius Measurement Unit
        "psi": CCO.ont00001694,   # Pounds Per Square Inch Measurement Unit
        "V":   CCO.ont00001450,   # Volt Measurement Unit
        "kPa": EX.Kilopascal,     # not in CCO -> our own unit
        "ohm": EX.Ohm,            # not in CCO -> our own unit
    }
    # sdc_kind -> SDC class (fall back to plain BFO SDC when CCO has no subclass)
    SDC_CLASSES = {
        "temperature": CCO.ont00000441,   # Temperature
        # pressure, voltage, resistance -> plain SDC (CCO's Pressure is a process profile, not a quality)
    }

    for row in reader:
        # IRIs can't contain spaces ("Chiller 3")
        art_id = row['artifact_id'].strip().replace(" ", "-")
        kind   = row['sdc_kind']
        stamp  = row['timestamp'].replace(":", "").replace("-", "")   # 20240317T190300Z

        # The four nodes in the pattern
        art  = EX[art_id]                              # ex:Boiler-07
        sdc  = EX[f"{art_id}_{kind}"]                  # ex:Boiler-07_temperature
        mice = EX[f"{art_id}_{kind}_{stamp}"]          # ex:Boiler-07_temperature_20240317T190300Z
        unit = UNITS[row['unit_label']]                # cco:ont00001724

        # rdf:type triples (instance -> class)
        g.add((art, RDF.type, CCO.ont00000995))                                # Material Artifact
        g.add((sdc, RDF.type, SDC_CLASSES.get(kind, OBO.BFO_0000020)))         # Temperature / Pressure / SDC
        g.add((mice, RDF.type, CCO.ont00001163))                               # MICE
        if unit.startswith(EX):
            g.add((unit, RDF.type, CCO.ont00000120))                           # Measurement Unit

        # Relations (instance -> instance / literal)
        g.add((art, OBO.BFO_0000196, sdc))                                     # bearer of
        g.add((mice, CCO.ont00001966, sdc))                                    # is a measurement of
        g.add((mice, CCO.ont00001863, unit))                                   # uses measurement unit
        g.add((mice, CCO.ont00001769, Literal(row['value'], datatype=XSD.decimal)))  # has decimal value


# 4. Import the CCO/BFO class hierarchy our data uses, so QC and SHACL can see it
cco = Graph().parse(SRC_DIR / "cco_merged.ttl", format="turtle")
ANNOTATIONS = (RDFS.label, SKOS.definition)

# CCO unit individuals are typed in CCO, not in our graph -> copy their types
for unit in UNITS.values():
    for t in cco.objects(unit, RDF.type):
        g.add((unit, RDF.type, t))
    for p in ANNOTATIONS:
        for o in cco.objects(unit, p):
            g.add((unit, p, o))

# Start from every class used in the data, then walk rdfs:subClassOf up to BFO
todo = [c for c in set(g.objects(None, RDF.type)) if c != OWL.NamedIndividual]
seen = set()
while todo:
    c = todo.pop()
    if c in seen:
        continue
    seen.add(c)
    g.add((c, RDF.type, OWL.Class))
    for p in ANNOTATIONS:
        for o in cco.objects(c, p):
            g.add((c, p, o))
    for parent in cco.objects(c, RDFS.subClassOf):
        if isinstance(parent, URIRef):      # skip blank-node restrictions
            g.add((c, RDFS.subClassOf, parent))
            todo.append(parent)

# 5. Serialize/Save the output to an RDF format (e.g., Turtle)
g.serialize(destination=SRC_DIR / "measure_cco.ttl", format='turtle')
print("Conversion complete! Saved to " + str(SRC_DIR / "measure_cco.ttl"))
