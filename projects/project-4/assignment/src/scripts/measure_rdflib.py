from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

# Define namespaces
CCO = Namespace("http://www.ontologyrepository.com/CommonCoreOntologies/")
EX = Namespace("http://example.org/")

# Create a new RDF graph
g = Graph()
g.bind("cco", CCO)
g.bind("ex", EX)

#Instantiate the Measurement Pattern
sample = EX.Sample001
temperature_quality = EX.Temperature001
measurement_process = EX.TemperatureMeasurement001
measurement_ice = EX.TemperatureMeasurementICE001
measurement_value = EX.TemperatureValue001
unit_celsius = EX.Celsius

# Declare the types (classes)
g.add((sample, RDF.type, CCO.MeasuredEntity))
g.add((temperature_quality, RDF.type, CCO.Temperature))
g.add((measurement_process, RDF.type, CCO.MeasurementProcess))
g.add((measurement_ice, RDF.type, CCO.MeasurementInformationContentEntity))
g.add((measurement_value, RDF.type, CCO.MeasurementValue))

# Link the entities following the CCO pattern
g.add((measurement_process, CCO.measures, temperature_quality))
g.add((temperature_quality, CCO.inheres_in, sample))
g.add((measurement_process, CCO.has_output, measurement_ice))
g.add((measurement_ice, CCO.denotes, measurement_value))
g.add((measurement_value, CCO.has_measurement_value, Literal(22.5, datatype=XSD.float)))
g.add((measurement_value, CCO.has_measurement_unit, unit_celsius))

#Serialize and view the RDF
print(g.serialize(format="turtle").decode())

