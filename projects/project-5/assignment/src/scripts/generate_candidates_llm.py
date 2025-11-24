#!/usr/bin/env python3
"""
Generate OWL 2 EL-compliant candidate axioms from enriched definitions.
Produces a Turtle file with formal axioms derived from natural language definitions.
"""

import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime

try:
    from rdflib import Graph, Namespace, URIRef, Literal, BNode
    from rdflib.namespace import RDF, RDFS, OWL, SKOS, DCTERMS
except ImportError:
    print("Error: rdflib is not installed. Please install it using:")
    print("  pip install rdflib")
    sys.exit(1)


# Define namespaces
CCO = Namespace("https://www.commoncoreontologies.org/")
CCO_ONT = Namespace("https://www.commoncoreontologies.org/ont")
FACILITY = Namespace("https://www.commoncoreontologies.org/FacilityOntology/")
OBO = Namespace("http://purl.obolibrary.org/obo/")


class OWL2ELAxiomGenerator:
    """Generates OWL 2 EL-compliant axioms from natural language definitions."""
    
    def __init__(self):
        """Initialize the axiom generator with parsing patterns and mappings."""
        
        # OWL 2 EL supports: SubClassOf, EquivalentClasses, DisjointClasses,
        # SubObjectPropertyOf, EquivalentObjectProperties, ObjectPropertyDomain,
        # ObjectPropertyRange, and existential restrictions
        
        # Pattern to extract parent class and characteristics
        self.canonical_pattern = re.compile(
            r'^(?:A|An)\s+(.+?)\s+that\s+(.+?)\.?$',
            re.IGNORECASE
        )
        
        # Patterns for extracting relationships
        self.relationship_patterns = [
            # "designed for/to" -> has_function/has_purpose
            (r'designed (?:for|to)\s+(.+)', 'has_function', True),
            # "used for/to" -> has_purpose
            (r'used (?:for|to)\s+(.+)', 'has_purpose', True),
            # "supports" -> supports
            (r'supports?\s+(.+)', 'supports', False),
            # "contains/containing" -> has_part
            (r'contains?\s+(.+)', 'has_part', False),
            (r'containing\s+(.+)', 'has_part', False),
            # "located in/at" -> located_in
            (r'located (?:in|at)\s+(.+)', 'located_in', False),
            # "provides" -> provides
            (r'provides?\s+(.+)', 'provides', False),
            # "facilitates" -> facilitates
            (r'facilitates?\s+(.+)', 'facilitates', False),
            # "serves" -> serves
            (r'serves?\s+(.+)', 'serves', False),
        ]
        
        # Known parent class mappings (from CCO)
        self.parent_class_map = {
            'Facility': CCO.ont00000192,
            'Healthcare Facility': CCO.ont00000055,
            'Religious Facility': CCO.ont00000052,
            'Educational Facility': CCO.ont00000270,
            'Transportation Facility': CCO.ont00000226,
            'Residential Facility': CCO.ont00000410,
            'Military Facility': CCO.ont00001052,
            'Commercial Facility': CCO.ont00001102,
            'Storage Facility': CCO.ont00000881,
            'Agricultural Facility': CCO.ont00000339,
            'Government Building': CCO.ont00000946,
            'Public Safety Facility': CCO.ont00000479,
            'Entertainment Facility': CCO.ont00000641,
            'Factory': CCO.ont00000782,
            'Mine': CCO.ont00000639,
            'Communications Facility': CCO.ont00000531,
            'Mailing Facility': CCO.ont00000655,
            'Product Transport Facility': CCO.ont00000677,
            'Electric Power Station': CCO.ont00001076,
            'Waste Management Facility': CCO.ont00001315,
            'Financial Facility': CCO.ont00001295,
            'Maintenance Facility': CCO.ont00001287,
            'Washing Facility': CCO.ont00001375,
            'Training Camp': CCO.ont00000332,
            'Water Treatment Facility': CCO.ont00000814,
        }
        
        # Object properties (OWL 2 EL compatible)
        self.object_properties = {
            'has_function': CCO.has_function,
            'has_purpose': CCO.has_purpose,
            'has_part': CCO.has_part,
            'part_of': CCO.part_of,
            'supports': CCO.supports,
            'located_in': CCO.located_in,
            'provides': CCO.provides,
            'facilitates': CCO.facilitates,
            'serves': CCO.serves,
            'designed_for': CCO.designed_for,
        }
    
    def parse_definition(self, definition: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse a canonical definition to extract parent class and characteristics.
        
        Args:
            definition: Enriched definition in canonical form
        
        Returns:
            Tuple of (parent_class, characteristics) or (None, None)
        """
        match = self.canonical_pattern.match(definition.strip())
        if match:
            parent_class = match.group(1).strip()
            characteristics = match.group(2).strip()
            return parent_class, characteristics
        return None, None
    
    def extract_relationships(self, characteristics: str) -> List[Tuple[str, str]]:
        """
        Extract relationships from the characteristics part of a definition.
        
        Args:
            characteristics: The "that..." part of the definition
        
        Returns:
            List of (property, object) tuples
        """
        relationships = []
        
        for pattern, prop_name, is_functional in self.relationship_patterns:
            matches = re.finditer(pattern, characteristics, re.IGNORECASE)
            for match in matches:
                obj = match.group(1).strip()
                # Clean up the object
                obj = re.sub(r'\s*\([^)]*\)', '', obj)  # Remove parenthetical notes
                obj = obj.rstrip('.,;')
                if obj:
                    relationships.append((prop_name, obj))
                    # For functional properties, only take the first match
                    if is_functional:
                        break
        
        return relationships
    
    def generate_class_axioms(self, iri: str, label: str, parent_class_str: str, 
                            characteristics: str, g: Graph) -> None:
        """
        Generate OWL 2 EL class axioms for an entity.
        
        Args:
            iri: The entity IRI
            label: The entity label
            parent_class_str: Parent class from definition
            characteristics: Characteristics from definition
            g: RDF graph to add axioms to
        """
        entity_uri = URIRef(iri)
        
        # Add class declaration
        g.add((entity_uri, RDF.type, OWL.Class))
        
        # Add label
        g.add((entity_uri, RDFS.label, Literal(label, lang='en')))
        
        # Add parent class (SubClassOf axiom)
        parent_uri = None
        if parent_class_str in self.parent_class_map:
            parent_uri = self.parent_class_map[parent_class_str]
        else:
            # Try to find partial matches
            for key, value in self.parent_class_map.items():
                if key.lower() in parent_class_str.lower() or parent_class_str.lower() in key.lower():
                    parent_uri = value
                    break
        
        if not parent_uri:
            # Default to general Facility
            parent_uri = self.parent_class_map.get('Facility', CCO.ont00000192)
        
        g.add((entity_uri, RDFS.subClassOf, parent_uri))
        
        # Extract and add relationships as restrictions (OWL 2 EL existential restrictions)
        relationships = self.extract_relationships(characteristics)
        
        for prop_name, obj_desc in relationships:
            if prop_name in self.object_properties:
                prop_uri = self.object_properties[prop_name]
                
                # Create an existential restriction (someValuesFrom)
                # In OWL 2 EL, we can use existential restrictions
                restriction = BNode()
                g.add((restriction, RDF.type, OWL.Restriction))
                g.add((restriction, OWL.onProperty, prop_uri))
                
                # For now, we'll create a named class for the object
                # In a real system, you'd want to resolve these to actual classes
                obj_class = self.create_object_class(obj_desc, g)
                g.add((restriction, OWL.someValuesFrom, obj_class))
                
                # Add the restriction as a superclass
                g.add((entity_uri, RDFS.subClassOf, restriction))
    
    def create_object_class(self, description: str, g: Graph) -> URIRef:
        """
        Create or reference a class for an object description.
        
        Args:
            description: Natural language description of the object
            g: RDF graph
        
        Returns:
            URI of the object class
        """
        # Simple heuristic: create a class based on key terms
        # In production, this would need more sophisticated NLP
        
        # Clean and normalize the description
        clean_desc = re.sub(r'[^\w\s]', '', description)
        tokens = clean_desc.split()
        
        # Look for known entity types
        known_types = {
            'worship': CCO.ReligiousActivity,
            'prayer': CCO.ReligiousActivity,
            'treatment': CCO.MedicalTreatment,
            'education': CCO.EducationalActivity,
            'learning': CCO.EducationalActivity,
            'storage': CCO.StorageFunction,
            'transportation': CCO.TransportationFunction,
            'military': CCO.MilitaryActivity,
            'commercial': CCO.CommercialActivity,
            'agricultural': CCO.AgriculturalActivity,
        }
        
        for token in tokens:
            token_lower = token.lower()
            for key, value in known_types.items():
                if key in token_lower:
                    return value
        
        # Default: create a generic class based on the description
        class_name = '_'.join(tokens[:3]) if len(tokens) >= 3 else '_'.join(tokens)
        class_uri = CCO[f"Activity_{class_name}"]
        
        # Add the class to the graph
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(description, lang='en')))
        
        return class_uri
    
    def generate_axioms(self, enriched_definitions: List[Dict[str, str]]) -> Graph:
        """
        Generate OWL 2 EL axioms from enriched definitions.
        
        Args:
            enriched_definitions: List of enriched definition dictionaries
        
        Returns:
            RDF graph containing the generated axioms
        """
        # Initialize graph with prefixes
        g = Graph()
        g.bind('', FACILITY)
        g.bind('cco', CCO)
        g.bind('owl', OWL)
        g.bind('rdf', RDF)
        g.bind('rdfs', RDFS)
        g.bind('skos', SKOS)
        g.bind('dcterms', DCTERMS)
        
        # Add ontology declaration
        onto_uri = URIRef("https://www.commoncoreontologies.org/FacilityOntology/CandidateAxioms")
        g.add((onto_uri, RDF.type, OWL.Ontology))
        g.add((onto_uri, RDFS.label, Literal("Facility Ontology Candidate Axioms", lang='en')))
        g.add((onto_uri, RDFS.comment, 
               Literal("OWL 2 EL-compliant candidate axioms generated from enriched definitions", lang='en')))
        g.add((onto_uri, DCTERMS.created, Literal(datetime.now().isoformat())))
        
        # Process each definition
        successful = 0
        failed = 0
        
        for defn in enriched_definitions:
            try:
                enriched = defn.get('enriched_definition', '')
                if not enriched:
                    continue
                
                # Parse the definition
                parent_class, characteristics = self.parse_definition(enriched)
                
                if parent_class and characteristics:
                    # Generate axioms
                    self.generate_class_axioms(
                        defn['IRI'],
                        defn['label'],
                        parent_class,
                        characteristics,
                        g
                    )
                    successful += 1
                else:
                    # Fall back to simple subclass axiom
                    entity_uri = URIRef(defn['IRI'])
                    g.add((entity_uri, RDF.type, OWL.Class))
                    g.add((entity_uri, RDFS.label, Literal(defn['label'], lang='en')))
                    g.add((entity_uri, RDFS.subClassOf, CCO.ont00000192))  # Default to Facility
                    g.add((entity_uri, SKOS.definition, Literal(enriched, lang='en')))
                    failed += 1
                    
            except Exception as e:
                print(f"  Warning: Failed to process {defn['label']}: {e}")
                failed += 1
        
        print(f"  Successfully generated axioms for {successful} definitions")
        print(f"  Failed or used fallback for {failed} definitions")
        
        return g
    
    def validate_owl2el_compliance(self, g: Graph) -> List[str]:
        """
        Validate that the generated axioms are OWL 2 EL compliant.
        
        Args:
            g: RDF graph to validate
        
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Check for OWL 2 EL restrictions
        # EL profile disallows: universal quantification, cardinality restrictions,
        # disjunction, negation, inverse properties, functional properties (in certain contexts)
        
        # Check for disallowed constructs
        disallowed = [
            (OWL.allValuesFrom, "Universal quantification (allValuesFrom)"),
            (OWL.maxCardinality, "Maximum cardinality restrictions"),
            (OWL.minCardinality, "Minimum cardinality restrictions"),
            (OWL.cardinality, "Exact cardinality restrictions"),
            (OWL.unionOf, "Union/disjunction"),
            (OWL.complementOf, "Negation/complement"),
            (OWL.inverseOf, "Inverse properties"),
        ]
        
        for construct, description in disallowed:
            if (None, construct, None) in g:
                warnings.append(f"Found {description} which is not allowed in OWL 2 EL")
        
        # Check that all restrictions are existential
        for restriction in g.subjects(RDF.type, OWL.Restriction):
            if (restriction, OWL.someValuesFrom, None) not in g:
                if (restriction, OWL.hasValue, None) not in g:
                    warnings.append(f"Restriction {restriction} is not an existential or hasValue restriction")
        
        return warnings


def main():
    """Main function to generate candidate axioms."""
    
    # Define file paths
    script_path = Path(__file__).resolve()
    
    # The script is in src/scripts/, need to go up one level to src/
    scripts_dir = script_path.parent
    src_dir = scripts_dir.parent
    
    # Define input and output paths
    input_file = src_dir / 'data' / 'definitions_enriched.csv'
    output_dir = src_dir / 'generated'
    output_file = output_dir / 'candidate_el.ttl'
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found!")
        print("Please run preprocess_definitions_llm.py first to generate the enriched definitions.")
        sys.exit(1)
    
    # Load enriched definitions
    print(f"Loading enriched definitions from {input_file}...")
    enriched_definitions = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            enriched_definitions.append(row)
    
    print(f"Loaded {len(enriched_definitions)} enriched definitions")
    
    # Initialize generator
    generator = OWL2ELAxiomGenerator()
    
    # Generate axioms
    print("\nGenerating OWL 2 EL axioms from definitions...")
    axiom_graph = generator.generate_axioms(enriched_definitions)
    
    # Validate OWL 2 EL compliance
    print("\nValidating OWL 2 EL compliance...")
    warnings = generator.validate_owl2el_compliance(axiom_graph)
    
    if warnings:
        print("  Validation warnings:")
        for warning in warnings:
            print(f"    - {warning}")
    else:
        print("  All axioms are OWL 2 EL compliant [OK]")
    
    # Save the axioms
    print(f"\nSaving candidate axioms to {output_file}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Serialize to Turtle format
    axiom_graph.serialize(destination=output_file, format='turtle')
    
    # Print statistics
    num_classes = len(list(axiom_graph.subjects(RDF.type, OWL.Class)))
    num_subclass = len(list(axiom_graph.subject_objects(RDFS.subClassOf)))
    num_restrictions = len(list(axiom_graph.subjects(RDF.type, OWL.Restriction)))
    
    print(f"\nGeneration complete!")
    print(f"  Classes defined: {num_classes}")
    print(f"  SubClassOf axioms: {num_subclass}")
    print(f"  Restrictions: {num_restrictions}")
    print(f"\nCandidate axioms saved to: {output_file}")
    
    # Show example axioms
    print("\nExample generated axioms (first 5 classes):")
    classes = list(axiom_graph.subjects(RDF.type, OWL.Class))[:5]
    for cls in classes:
        label = axiom_graph.value(cls, RDFS.label)
        parent = axiom_graph.value(cls, RDFS.subClassOf)
        if label:
            parent_label = axiom_graph.value(parent, RDFS.label) if parent else "Unknown"
            print(f"  {label} SubClassOf {parent_label if parent_label else parent}")


if __name__ == "__main__":
    main()

    