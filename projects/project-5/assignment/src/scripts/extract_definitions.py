#!/usr/bin/env python3
"""
Extract textual definitions from FacilityOntology.ttl
Outputs a CSV file with IRI, label, and definition columns.

Requirements:
    pip install rdflib
"""

import csv
import sys
from pathlib import Path

try:
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, RDFS, OWL, SKOS, DCTERMS
except ImportError:
    print("Error: rdflib is not installed. Please install it using:")
    print("  pip install rdflib")
    sys.exit(1)

# Define common namespaces for definition properties
DC = Namespace("http://purl.org/dc/elements/1.1/")
DCT = Namespace("http://purl.org/dc/terms/")
IAO = Namespace("http://purl.obolibrary.org/obo/IAO_")  # Information Artifact Ontology
OBO = Namespace("http://purl.obolibrary.org/obo/")
CCO = Namespace("https://www.commoncoreontologies.org/")

def extract_definitions(input_file: Path, output_file: Path) -> None:
    """
    Extract definitions from an RDF/TTL file and save to CSV.
    
    Args:
        input_file: Path to the input TTL file
        output_file: Path to the output CSV file
    """
    
    # Load the ontology
    print(f"Loading ontology from {input_file}...")
    g = Graph()
    try:
        g.parse(input_file, format='turtle')
        print(f"Successfully loaded {len(g)} triples")
    except Exception as e:
        print(f"Error loading ontology: {e}")
        sys.exit(1)
    
    # Common properties used for definitions
    definition_predicates = [
        SKOS.definition,
        DCTERMS.description,
        DC.description,
        RDFS.comment,
        URIRef("http://purl.obolibrary.org/obo/IAO_0000115"),  # IAO definition
        URIRef("http://www.w3.org/2004/02/skos/core#definition"),
        URIRef("http://purl.org/dc/elements/1.1/description"),
        URIRef("http://purl.org/dc/terms/description"),
        URIRef("http://www.w3.org/2000/01/rdf-schema#comment"),
    ]
    
    # Common properties used for labels
    label_predicates = [
        RDFS.label,
        SKOS.prefLabel,
        SKOS.altLabel,
        DCTERMS.title,
        DC.title,
        URIRef("http://www.w3.org/2004/02/skos/core#prefLabel"),
        URIRef("http://www.w3.org/2004/02/skos/core#altLabel"),
        URIRef("http://purl.org/dc/elements/1.1/title"),
        URIRef("http://purl.org/dc/terms/title"),
    ]
    
    # Collect all entities (classes and individuals)
    entities = set()
    
    # Find all classes
    for cls in g.subjects(RDF.type, OWL.Class):
        entities.add(cls)
    
    # Find all named individuals
    for ind in g.subjects(RDF.type, OWL.NamedIndividual):
        entities.add(ind)
    
    # Also include any subject that has a label or definition
    for pred in label_predicates + definition_predicates:
        for subj in g.subjects(pred, None):
            if isinstance(subj, URIRef):
                entities.add(subj)
    
    print(f"Found {len(entities)} entities to process")
    
    # Extract definitions
    definitions_data = []
    
    for entity in entities:
        if not isinstance(entity, URIRef):
            continue
            
        # Get IRI
        iri = str(entity)
        
        # Get label (try multiple predicates)
        label = None
        for label_pred in label_predicates:
            labels = list(g.objects(entity, label_pred))
            if labels:
                # Prefer English labels if language-tagged
                en_labels = [l for l in labels if isinstance(l, Literal) and (l.language == 'en' or l.language is None)]
                if en_labels:
                    label = str(en_labels[0])
                else:
                    label = str(labels[0])
                break
        
        # If no label found, use the local name from the IRI
        if not label:
            if '#' in iri:
                label = iri.split('#')[-1]
            else:
                label = iri.split('/')[-1]
        
        # Get definition (try multiple predicates)
        definition = None
        for def_pred in definition_predicates:
            definitions = list(g.objects(entity, def_pred))
            if definitions:
                # Prefer English definitions if language-tagged
                en_defs = [d for d in definitions if isinstance(d, Literal) and (d.language == 'en' or d.language is None)]
                if en_defs:
                    # For skos:definition, prefer it over others
                    if def_pred == SKOS.definition or def_pred == URIRef("http://www.w3.org/2004/02/skos/core#definition"):
                        definition = str(en_defs[0])
                        break
                    elif not definition:  # Use other definition types if no skos:definition found yet
                        definition = str(en_defs[0])
                elif not definition:
                    definition = str(definitions[0])
        
        # Only include entries that have at least a label
        if label:
            definitions_data.append({
                'IRI': iri,
                'label': label,
                'definition': definition if definition else ''
            })
    
    # Sort by label for better readability
    definitions_data.sort(key=lambda x: x['label'].lower())
    
    # Write to CSV
    print(f"Writing {len(definitions_data)} definitions to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['IRI', 'label', 'definition']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(definitions_data)
    
    print(f"Successfully extracted {len(definitions_data)} definitions")
    
    # Print summary statistics
    with_defs = sum(1 for d in definitions_data if d['definition'])
    without_defs = len(definitions_data) - with_defs
    print(f"  - Entities with definitions: {with_defs}")
    print(f"  - Entities without definitions: {without_defs}")


def main():
    """Main function to run the extraction script.
    
    Expected directory structure (based on your paths):
    src/
    ├── scripts/
    │   └── extract_definitions.py (this script)
    ├── FacilityOntology.ttl (input)
    └── data/
        └── definitions.csv (output will be created here)
    """
    
    # Define file paths
    script_path = Path(__file__).resolve()
    
    # The script is in src/scripts/, need to go up one level to src/
    scripts_dir = script_path.parent
    src_dir = scripts_dir.parent
    
    # Define input and output paths
    input_file = src_dir / 'FacilityOntology.ttl'
    output_file = src_dir / 'data' / 'definitions.csv'
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found!")
        print(f"Please ensure the FacilityOntology.ttl file is in the 'src' directory.")
        sys.exit(1)
    
    # Extract definitions
    extract_definitions(input_file, output_file)
    
    print(f"\nDefinitions successfully extracted to: {output_file}")


if __name__ == "__main__":
    main()

    