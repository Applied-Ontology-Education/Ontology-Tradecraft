from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD
from datetime import datetime
import uuid

# Setup Namespaces
CCO = Namespace("https://www.commoncoreontologies.org/")
BFO = Namespace("http://purl.obolibrary.org/obo/bfo.owl") 
DATA = Namespace("http://example.org/sensor-data/")    #madeup namespace name for my sensor data
 
def load_cco_ontology(cco_path):
    cco_graph = Graph()
    cco_graph.parse(cco_path, format="turtle")
    return cco_graph

print(Literal('212.0', datatype=XSD.decimal))
