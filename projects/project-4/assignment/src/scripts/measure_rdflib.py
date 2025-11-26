from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD
from datetime import datetime

# Step 1: Create a new RDF graph (think of it as a container for your data)
g = Graph()

# Step 2: Define namespaces (these are like shortcuts for long URIs)
# CCO and BFO namespaces with exact IRIs
CCO = Namespace("https://www.commoncoreontologies.org/")
BFO = Namespace("http://purl.obolibrary.org/obo/")  # Basic Formal Ontology

# Your custom namespace for instances (actual data)
EX = Namespace("http://example.org/measurement/")

# Bind namespaces to prefixes (makes the output file more readable)
g.bind("cco", CCO)
g.bind("bfo", BFO)
g.bind("ex", EX)
g.bind("xsd", XSD)

# Define the EXACT IRIs required by the validation test
# These are the specific class and property IRIs from CCO/BFO
IRI_SDC = URIRef("http://purl.obolibrary.org/obo/BFO_0000020")  # Specifically Dependent Continuant
IRI_ART = URIRef("https://www.commoncoreontologies.org/ont00000995")  # Artifact
IRI_MU = URIRef("https://www.commoncoreontologies.org/ont00000120")  # Measurement Unit
IRI_MICE = URIRef("https://www.commoncoreontologies.org/ont00001163")  # Measurement Information Content Entity

IRI_BEARER_OF = URIRef("http://purl.obolibrary.org/obo/BFO_0000196")  # bearer of
IRI_IS_MEASURE_OF = URIRef("https://www.commoncoreontologies.org/ont00001966")  # is measure of
IRI_USES_MU = URIRef("https://www.commoncoreontologies.org/ont00001863")  # uses measurement unit


# Example 1: Temperature Measurement
# -----------------------------------
print("Example 1: Temperature Measurement")

# 1. Create the Artifact (measurement instrument)
thermometer = EX["thermometer-001"]
g.add((thermometer, RDF.type, IRI_ART))
g.add((thermometer, RDFS.label, Literal("Digital Thermometer TH-200")))

# 2. Create the SDC (Specifically Dependent Continuant - the temperature quality)
temperature_sdc = EX["temperature-sdc-001"]
g.add((temperature_sdc, RDF.type, IRI_SDC))
g.add((temperature_sdc, RDFS.label, Literal("Temperature of Room A")))

# 3. Link: Artifact is "bearer of" SDC (using exact IRI)
g.add((thermometer, IRI_BEARER_OF, temperature_sdc))

# 4. Create the MICE (Measurement Information Content Entity)
temp_mice = EX["temp-measurement-001"]
g.add((temp_mice, RDF.type, IRI_MICE))
g.add((temp_mice, RDFS.label, Literal("Temperature Measurement: 22.5°C")))

# 5. Link: MICE "is measure of" SDC (using exact IRI)
g.add((temp_mice, IRI_IS_MEASURE_OF, temperature_sdc))

# 6. Create and link the Measurement Unit (using exact IRI)
celsius_unit = EX["celsius-unit"]
g.add((celsius_unit, RDF.type, IRI_MU))
g.add((celsius_unit, RDFS.label, Literal("Degree Celsius")))
g.add((temp_mice, IRI_USES_MU, celsius_unit))

print("  ✓ Artifact: thermometer-001")
print("  ✓ SDC: temperature-sdc-001 (Temperature)")
print("  ✓ MICE: temp-measurement-001 (22.5)")
print("  ✓ Unit: celsius-unit (°C)\n")


# Example 2: Length Measurement
# ------------------------------
print("Example 2: Length Measurement")

# 1. Artifact (measuring tape)
measuring_tape = EX["measuring-tape-001"]
g.add((measuring_tape, RDF.type, IRI_ART))
g.add((measuring_tape, RDFS.label, Literal("Measuring Tape MT-5000")))

# 2. SDC (length quality)
length_sdc = EX["length-sdc-001"]
g.add((length_sdc, RDF.type, IRI_SDC))
g.add((length_sdc, RDFS.label, Literal("Length of Table")))

# 3. Link: Artifact is bearer of SDC (using exact IRI)
g.add((measuring_tape, IRI_BEARER_OF, length_sdc))

# 4. MICE
length_mice = EX["length-measurement-001"]
g.add((length_mice, RDF.type, IRI_MICE))
g.add((length_mice, RDFS.label, Literal("Length Measurement: 1.5m")))

# 5. Link: MICE is measure of SDC (using exact IRI)
g.add((length_mice, IRI_IS_MEASURE_OF, length_sdc))

# 6. Unit (using exact IRI)
meter_unit = EX["meter-unit"]
g.add((meter_unit, RDF.type, IRI_MU))
g.add((meter_unit, RDFS.label, Literal("Meter")))
g.add((length_mice, IRI_USES_MU, meter_unit))

print("  ✓ Artifact: measuring-tape-001")
print("  ✓ SDC: length-sdc-001 (Length)")
print("  ✓ MICE: length-measurement-001 (1.5)")
print("  ✓ Unit: meter-unit (m)\n")


# Example 3: Mass Measurement
# ----------------------------
print("Example 3: Mass Measurement")

# 1. Artifact (scale)
scale = EX["scale-001"]
g.add((scale, RDF.type, IRI_ART))
g.add((scale, RDFS.label, Literal("Digital Scale DS-100")))

# 2. SDC (mass quality)
mass_sdc = EX["mass-sdc-001"]
g.add((mass_sdc, RDF.type, IRI_SDC))
g.add((mass_sdc, RDFS.label, Literal("Mass of Object")))

# 3. Link: Artifact is bearer of SDC (using exact IRI)
g.add((scale, IRI_BEARER_OF, mass_sdc))

# 4. MICE
mass_mice = EX["mass-measurement-001"]
g.add((mass_mice, RDF.type, IRI_MICE))
g.add((mass_mice, RDFS.label, Literal("Mass Measurement: 5.2kg")))

# 5. Link: MICE is measure of SDC (using exact IRI)
g.add((mass_mice, IRI_IS_MEASURE_OF, mass_sdc))

# 6. Unit (using exact IRI)
kilogram_unit = EX["kilogram-unit"]
g.add((kilogram_unit, RDF.type, IRI_MU))
g.add((kilogram_unit, RDFS.label, Literal("Kilogram")))
g.add((mass_mice, IRI_USES_MU, kilogram_unit))

print("  ✓ Artifact: scale-001")
print("  ✓ SDC: mass-sdc-001 (Mass)")
print("  ✓ MICE: mass-measurement-001 (5.2)")
print("  ✓ Unit: kilogram-unit (kg)\n")


# Step 4: Save the graph to a file
# =================================

output_file = "../measure_cco.ttl"

# Serialize to Turtle format (a human-readable RDF format)
g.serialize(destination=output_file, format="turtle")

print("\n" + "="*60)
print("✅ RDF graph created successfully!")
print("="*60)
print(f"📁 Output saved to: {output_file}")
print(f"📊 Total triples: {len(g)}")

