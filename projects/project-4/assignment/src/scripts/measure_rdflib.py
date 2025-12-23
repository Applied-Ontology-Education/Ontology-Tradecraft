import pandas as pd
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


def transform_csv_to_ttl(csv_path, output_path, cco_ontology_path):
    """
    Transform sensor CSV data into RDF following CCO/BFO patterns.
    
    Expected CSV columns:
    - artifact_id: ID of the artifact/device being measured
    - timestamp: ISO 8601 timestamp of measurement
    - sdc_kind: Specifically Dependent Continuant kind (measurement type)
    - unit_label: Unit of measurement (e.g., 'celsius', 'meters')
    - value: Numeric value of the measurement
    """
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Initialize the RDF Graph
    g = Graph()
    g.bind("cco", CCO)
    g.bind("bfo", BFO)
    g.bind("data", DATA)
    g.bind("xsd", XSD)
    
    # Load CCO ontology for validation
    cco_graph = load_cco_ontology(cco_ontology_path)
    print(f"Loaded CCO ontology from {cco_ontology_path}")
    
    # Track created entities to avoid duplicates
    artifacts_created = set()
    units_created = set()
    
    for index, row in df.iterrows():
        # Generate unique URI for this measurement observation
        measurement_id = str(uuid.uuid4())
        
        # 1. ARTIFACT INSTANCE (art-inst) - the thing being measured
        artifact_uri = DATA[f"artifact_{row['artifact_id']}"]
        if artifact_uri not in artifacts_created:
            g.add((artifact_uri, RDF.type, CCO['ont00000995']))
            artifacts_created.add(artifact_uri)
        
        # 2. SDC INSTANCE (sdc-inst) - Specifically Dependent Continuant (Quality)
        # This quality inheres in this artifact
        sdc_uri = DATA[f"quality_{row['sdc_kind']}_{measurement_id}"]
        g.add((sdc_uri, RDF.type, BFO['BFO_0000020']))  # BFO:0000020 = Specifically Dependent Continuant
        
        # Relationship: sdc-inst BFO:inheres_in art-inst (BFO:0000052)
        g.add((sdc_uri, BFO['BFO_0000052'], artifact_uri))
        
        # 3. MICE INSTANCE (mice-inst) - Measurement Information Content Entity
        mice_uri = DATA[f"measurement_data_{measurement_id}"]
        g.add((mice_uri, RDF.type, CCO['ont00001163']))
        
        # Relationship: mice-inst CCO:is_a_measurement_of sdc-inst
        g.add((mice_uri, CCO['ont00001966'], sdc_uri))
        
        # 4. VALUE - the actual measurement value
        # Relationship: mice-inst CCO:has_value xsd:value
        # this MICE has decimal value such and such
        g.add((mice_uri, CCO['ont00001769'], 
               Literal(row['value'], datatype=XSD.decimal)))
        
        # 5. UNIT INSTANCE (unit-inst) - Measurement Unit
        # this entity is a measurement unit
        unit_uri = DATA[f"unit_{row['unit_label']}"]
        if unit_uri not in units_created:
            g.add((unit_uri, RDF.type, CCO['ont00000120']))
            units_created.add(unit_uri)
        
        # Relationship: mice-inst CCO:uses_measurement_unit unit-inst
        g.add((mice_uri, CCO['ont00001863'], unit_uri))
        
        # 6. TIMESTAMP - associate with the measurement ICE
        # this MICE has datetime value such and such
        try:
            timestamp_dt = pd.to_datetime(row['timestamp'])   
            timestamp_literal = Literal(timestamp_dt.isoformat(), datatype=XSD.dateTime)
            print(timestamp_literal)    #no need of .isoformat() method here
            # You'll need to replace 'has_timestamp' with the actual CCO ID
            g.add((mice_uri, CCO['ont00001767'], timestamp_literal))
        except Exception as e:
            print(f"Warning: Could not parse timestamp '{row['timestamp']}' at row {index}: {e}")
    

    # Save to Turtle format
    g.serialize(destination=output_path, format="turtle")
    print(f"Success! Transformed {len(df)} rows into RDF.")
    print(f"Created {len(artifacts_created)} artifacts, {len(units_created)} units")
    print(f"RDF graph saved to {output_path}")
    return g

# Run the transformation
if __name__ == "__main__":
    csv_path = r"C:\Users\Federico\Code-Projects\FedeDon-Github-Repos\Ontology-Tradecraft\projects\project-4\assignment\src\data\readings_normalized.csv"
    output_path = r"C:\Users\Federico\Code-Projects\FedeDon-Github-Repos\Ontology-Tradecraft\projects\project-4\assignment\src\measure_cco.ttl"
    
    # Provide path to your local CCO.ttl file
    cco_path = r"C:\Users\Federico\Code-Projects\FedeDon-Github-Repos\Ontology-Tradecraft\projects\project-4\assignment\src\cco_merged.ttl"   
    
    g = transform_csv_to_ttl(csv_path, output_path, cco_path)
    
    # Optional: Print some statistics
"""     print(f"\nTotal triples: {len(g)}")
    print(f"Sample triples:")
    for i, (s, p, o) in enumerate(g):
        if i < 5:
            print(f"  {s} -> {p} -> {o}")
        else:
            break """