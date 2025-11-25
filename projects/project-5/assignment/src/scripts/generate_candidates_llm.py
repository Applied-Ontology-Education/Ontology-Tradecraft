#!/usr/bin/env python3
"""
Generate OWL 2 EL-compliant candidate axioms from enriched definitions using Claude.
Produces a Turtle file with formal axioms derived from natural language definitions.
"""

import csv
import json
import os
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

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Error: anthropic package not installed. Install with: pip install anthropic")
    sys.exit(1)


# Define namespaces
CCO = Namespace("https://www.commoncoreontologies.org/")
CCO_ONT = Namespace("https://www.commoncoreontologies.org/ont")
FACILITY = Namespace("https://www.commoncoreontologies.org/FacilityOntology/")
OBO = Namespace("http://purl.obolibrary.org/obo/")


class OWL2ELAxiomGenerator:
    """Generates OWL 2 EL-compliant axioms from natural language definitions using Claude."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the axiom generator with Claude API and parsing patterns."""
        
        if not ANTHROPIC_AVAILABLE:
            raise ValueError("Anthropic package not installed. Install with: pip install anthropic")
        
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = "claude-sonnet-4-20250514"
        
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
    
    def parse_definition_with_claude(self, label: str, definition: str, iri: str) -> Dict:
        """
        Use Claude to parse a definition and extract structured axiom information.
        
        Args:
            label: Entity label
            definition: Enriched definition
            iri: Entity IRI
        
        Returns:
            Dictionary with parsed axiom information
        """
        parent_classes_str = ', '.join(self.parent_class_map.keys())
        properties_str = ', '.join(self.object_properties.keys())
        
        prompt = f"""You are an ontology engineering expert. Parse the following definition into structured axiom components for OWL 2 EL.

Entity Label: {label}
Entity IRI: {iri}
Definition: {definition}

Available Parent Classes:
{parent_classes_str}

Available Object Properties:
{properties_str}

Task: Extract the following information from the definition:
1. Parent Class: The most specific parent class from the available list (choose the closest match)
2. Relationships: List of (property, object_description) pairs that describe characteristics

Return ONLY a valid JSON object with this exact structure:
{{
  "parent_class": "exact parent class name from available list",
  "relationships": [
    {{"property": "property_name", "object": "description of what the property relates to"}}
  ]
}}

Rules:
- parent_class must be from the available parent classes list (exact match)
- If no exact match, use "Facility" as the default parent
- property must be from the available object properties list (exact match)
- object should be a brief description (2-5 words) of what the relationship connects to
- Extract all relevant relationships from the definition
- Return ONLY the JSON, no other text"""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = message.content[0].text.strip()
            
            # Remove markdown code blocks if present
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            response_text = response_text.strip()
            
            # Parse JSON
            parsed = json.loads(response_text)
            
            # Validate structure
            if 'parent_class' not in parsed or 'relationships' not in parsed:
                raise ValueError("Missing required fields in response")
            
            return parsed
            
        except Exception as e:
            print(f"  Warning: Claude parsing failed for '{label}': {e}")
            # Return default structure
            return {
                'parent_class': 'Facility',
                'relationships': []
            }
    
    def generate_class_axioms(self, iri: str, label: str, definition: str, g: Graph) -> None:
        """
        Generate OWL 2 EL class axioms for an entity using Claude for parsing.
        
        Args:
            iri: The entity IRI
            label: The entity label
            definition: The enriched definition
            g: RDF graph to add axioms to
        """
        entity_uri = URIRef(iri)
        
        # Add class declaration
        g.add((entity_uri, RDF.type, OWL.Class))
        
        # Add label
        g.add((entity_uri, RDFS.label, Literal(label, lang='en')))
        
        # Add definition as skos:definition
        g.add((entity_uri, SKOS.definition, Literal(definition, lang='en')))
        
        # Use Claude to parse the definition
        parsed = self.parse_definition_with_claude(label, definition, iri)
        
        # Add parent class (SubClassOf axiom)
        parent_class_str = parsed.get('parent_class', 'Facility')
        parent_uri = self.parent_class_map.get(parent_class_str)
        
        if not parent_uri:
            # Try to find partial matches
            for key, value in self.parent_class_map.items():
                if key.lower() in parent_class_str.lower() or parent_class_str.lower() in key.lower():
                    parent_uri = value
                    break
        
        if not parent_uri:
            # Default to general Facility
            parent_uri = self.parent_class_map.get('Facility', CCO.ont00000192)
        
        g.add((entity_uri, RDFS.subClassOf, parent_uri))
        
        # Skip adding relationship restrictions for flat candidate output

    
    def create_object_class(self, description: str, g: Graph) -> URIRef:
        """
        Create or reference a class for an object description.
        
        Args:
            description: Natural language description of the object
            g: RDF graph
        
        Returns:
            URI of the object class
        """
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
        Generate OWL 2 EL axioms from enriched definitions using Claude.
        
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
               Literal("OWL 2 EL-compliant candidate axioms generated from enriched definitions using Claude", lang='en')))
        g.add((onto_uri, DCTERMS.created, Literal(datetime.now().isoformat())))
        
        # Process each definition
        successful = 0
        failed = 0
        total = len(enriched_definitions)
        
        print(f"Processing {total} definitions with Claude...")
        
        for i, defn in enumerate(enriched_definitions, 1):
            try:
                enriched = defn.get('enriched_definition', '')
                if not enriched:
                    enriched = defn.get('original_definition', '')
                
                if not enriched:
                    print(f"  [{i}/{total}] Skipping {defn['label']}: No definition available")
                    failed += 1
                    continue
                
                print(f"  [{i}/{total}] Processing {defn['label']}...")
                
                # Generate axioms using Claude
                self.generate_class_axioms(
                    defn['IRI'],
                    defn['label'],
                    enriched,
                    g
                )
                successful += 1
                
                # Rate limiting
                if i % 10 == 0:
                    import time
                    time.sleep(1)
                    
            except Exception as e:
                print(f"  Warning: Failed to process {defn['label']}: {e}")
                # Add basic axioms as fallback
                entity_uri = URIRef(defn['IRI'])
                g.add((entity_uri, RDF.type, OWL.Class))
                g.add((entity_uri, RDFS.label, Literal(defn['label'], lang='en')))
                g.add((entity_uri, RDFS.subClassOf, CCO.ont00000192))  # Default to Facility
                failed += 1
        
        print(f"\n  Successfully generated axioms for {successful} definitions")
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
    
    # Check for Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("Error: No Anthropic API key found!")
        print("Set the environment variable:")
        print("  PowerShell: $env:ANTHROPIC_API_KEY = 'your-key-here'")
        print("  Linux/Mac: export ANTHROPIC_API_KEY='your-key-here'")
        print("\nGet your API key from: https://console.anthropic.com/")
        sys.exit(1)
    
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
    print("\nInitializing Claude-based axiom generator...")
    generator = OWL2ELAxiomGenerator(api_key=api_key)
    
    # Generate axioms
    print("\nGenerating OWL 2 EL axioms using Claude...")
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
        if label and not isinstance(parent, BNode):
            parent_label = axiom_graph.value(parent, RDFS.label) if parent else "Unknown"
            print(f"  {label} SubClassOf {parent_label if parent_label else parent}")


if __name__ == "__main__":
    main()

    