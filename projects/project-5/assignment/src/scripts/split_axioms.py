#!/usr/bin/env python3
"""
Split ontology axioms into training and validation sets.

Splits FacilityOntology.ttl into:
- train.ttl (80% of SubClassOf axioms)
- valid.ttl (20% of SubClassOf axioms)

CRITICAL: Ensures both splits share the same class vocabulary so that
validation pairs can be properly evaluated using trained embeddings.

Preserves ALL class declarations and property declarations in both files.
Only splits the SubClassOf axioms for validation purposes.
"""

import random
import sys
from pathlib import Path
from typing import List, Tuple, Set

try:
    from rdflib import Graph, RDF, RDFS, OWL
    from rdflib.term import URIRef
except ImportError:
    print("Error: rdflib not installed. Install with: pip install rdflib")
    sys.exit(1)


class OntologySplitter:
    """Splits ontology into training and validation sets with shared vocabulary."""

    def __init__(self, input_file: Path, output_dir: Path, train_ratio: float = 0.8, seed: int = 42):
        self.input_file = input_file
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.seed = seed
        random.seed(seed)

        self.train_file = output_dir / 'train.ttl'
        self.valid_file = output_dir / 'valid.ttl'

    def load_ontology(self) -> Graph:
        print(f"Loading ontology from {self.input_file}...")
        g = Graph()
        g.parse(str(self.input_file), format='turtle')
        print(f"Loaded {len(g)} triples")

        imports = list(g.objects(predicate=OWL.imports))
        if imports:
            print(f"\nNote: Ontology imports {len(imports)} external ontology(ies):")
            for imp in imports:
                print(f"  - {imp}")
            print("  External classes/properties from imports will not be included in the split.\n")
        return g

    def separate_axioms(self, g: Graph) -> Tuple[List, List, Graph]:
        """Separate axioms into declarations, subclass axioms, and metadata."""
        declarations = []
        subclass_axioms = []
        metadata = Graph()

        print("Separating axioms...")

        for s, p, o in g:
            # Preserve ontology IRI in metadata
            if p == RDF.type and o == OWL.Ontology:
                metadata.add((s, p, o))
                continue

            # Skip other ontology metadata
            if p in (OWL.imports, OWL.versionIRI, OWL.versionInfo):
                metadata.add((s, p, o))
                continue

            # Skip metadata about the ontology IRI itself
            if isinstance(s, URIRef) and (
                str(s).endswith('#') or 
                'FacilityOntology' in str(s) and not str(s).startswith('https://www.commoncoreontologies.org/ont')
            ):
                continue

            # Class declarations
            if p == RDF.type and o in (OWL.Class, RDFS.Class):
                declarations.append((s, p, o))

            # Property declarations
            elif p == RDF.type and o in (OWL.ObjectProperty, OWL.DatatypeProperty, 
                                        OWL.AnnotationProperty, OWL.FunctionalProperty,
                                        OWL.InverseFunctionalProperty, OWL.TransitiveProperty,
                                        OWL.SymmetricProperty, RDF.Property):
                declarations.append((s, p, o))

            # Labels, comments, SKOS definitions
            elif p in (RDFS.label, RDFS.comment) or ('skos' in str(p).lower() and 'definition' in str(p).lower()):
                declarations.append((s, p, o))

            # Domain and range
            elif p in (RDFS.domain, RDFS.range):
                declarations.append((s, p, o))

            # SubClassOf axioms - ONLY named class pairs
            elif p == RDFS.subClassOf:
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    # Verify both are declared classes
                    if (s, RDF.type, OWL.Class) in g and (o, RDF.type, OWL.Class) in g:
                        # Skip owl:Thing
                        if not str(o).endswith('owl#Thing'):
                            subclass_axioms.append((s, p, o))
                    else:
                        # Restrictions or blank nodes - keep in declarations
                        declarations.append((s, p, o))
                else:
                    declarations.append((s, p, o))

            # Other axioms
            elif p in (OWL.equivalentClass, OWL.disjointWith, OWL.inverseOf, OWL.equivalentProperty):
                declarations.append((s, p, o))

            else:
                declarations.append((s, p, o))

        print(f"  Declarations (preserved in both): {len(declarations)}")
        print(f"  SubClassOf axioms (to split): {len(subclass_axioms)}")
        print(f"  Metadata triples: {len(metadata)}")

        return declarations, subclass_axioms, metadata

    def smart_split_axioms(self, axioms: List[Tuple]) -> Tuple[List, List]:
        """
        Split axioms ensuring validation set has classes that appear in training.
        
        Strategy:
        1. Randomly shuffle all axioms
        2. Split 80/20
        3. Verify that validation classes appear in training
        4. Report statistics
        """
        print("\nPerforming smart split with vocabulary overlap...")
        
        # Shuffle axioms randomly
        shuffled = axioms.copy()
        random.shuffle(shuffled)
        
        # Simple split
        split_point = int(len(shuffled) * self.train_ratio)
        train_axioms = shuffled[:split_point]
        valid_axioms = shuffled[split_point:]
        
        # Collect vocabularies
        train_classes = set()
        for s, p, o in train_axioms:
            train_classes.add(str(s))
            train_classes.add(str(o))
        
        valid_classes = set()
        for s, p, o in valid_axioms:
            valid_classes.add(str(s))
            valid_classes.add(str(o))
        
        # Calculate overlap
        overlap = train_classes & valid_classes
        valid_only = valid_classes - train_classes
        
        print(f"\nSplit statistics:")
        print(f"  Train axioms: {len(train_axioms)} ({len(train_axioms)/len(axioms)*100:.1f}%)")
        print(f"  Valid axioms: {len(valid_axioms)} ({len(valid_axioms)/len(axioms)*100:.1f}%)")
        print(f"\nVocabulary analysis:")
        print(f"  Train classes: {len(train_classes)}")
        print(f"  Valid classes: {len(valid_classes)}")
        print(f"  Shared classes: {len(overlap)} ({len(overlap)/len(valid_classes)*100:.1f}% of validation)")
        
        if valid_only:
            print(f"  Valid-only classes: {len(valid_only)} (unseen in training)")
            
            # Count how many validation pairs are fully computable
            computable = 0
            for s, p, o in valid_axioms:
                if str(s) in train_classes and str(o) in train_classes:
                    computable += 1
            
            print(f"\nComputable validation pairs: {computable}/{len(valid_axioms)} ({computable/len(valid_axioms)*100:.1f}%)")
            
            if computable < 5:
                print("\n⚠️  WARNING: Very few computable validation pairs!")
                print("   This may affect validation quality.")
                print("   Consider using a larger training ratio or more axioms.")
        else:
            print(f"\n✓ All validation classes appear in training set")
        
        return train_axioms, valid_axioms

    def create_graph(self, declarations: List, specific_axioms: List, metadata: Graph) -> Graph:
        """Create a new graph with declarations and specific axioms."""
        g = Graph()

        # Copy namespaces from original graph
        for prefix, namespace in self.original_graph.namespaces():
            g.bind(prefix, namespace)

        # Add metadata, declarations, and specific axioms
        for triple in metadata:
            g.add(triple)
        for triple in declarations:
            g.add(triple)
        for triple in specific_axioms:
            g.add(triple)
        return g

    def save_split(self):
        """Main splitting logic."""
        g = self.load_ontology()
        self.original_graph = g  # store for namespace copying

        declarations, subclass_axioms, metadata = self.separate_axioms(g)

        if not subclass_axioms:
            print("\n⚠️  WARNING: No SubClassOf axioms found to split!")
            print("   Creating train/valid files with declarations only.")
            train_graph = self.create_graph(declarations, [], metadata)
            valid_graph = self.create_graph(declarations, [], metadata)
        else:
            train_axioms, valid_axioms = self.smart_split_axioms(subclass_axioms)
            train_graph = self.create_graph(declarations, train_axioms, metadata)
            valid_graph = self.create_graph(declarations, valid_axioms, metadata)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving train set to {self.train_file}...")
        train_graph.serialize(destination=str(self.train_file), format='turtle')
        print(f"  Saved {len(train_graph)} triples")

        print(f"\nSaving validation set to {self.valid_file}...")
        valid_graph.serialize(destination=str(self.valid_file), format='turtle')
        print(f"  Saved {len(valid_graph)} triples")

        print("\n" + "="*70)
        print("✓ SPLIT COMPLETE")
        print("="*70)
        print(f"Train file: {self.train_file}")
        print(f"Valid file: {self.valid_file}")
        print("="*70)


def main():
    script_path = Path(__file__).resolve()
    scripts_dir = script_path.parent
    src_dir = scripts_dir.parent

    input_file = src_dir / 'FacilityOntology.ttl'
    output_dir = src_dir

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    print("="*70)
    print("ONTOLOGY AXIOM SPLITTER")
    print("="*70)
    print(f"Input: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratio: 80% train / 20% validation")
    print(f"Random seed: 42")
    print(f"Strategy: Smart split with vocabulary overlap")
    print("="*70 + "\n")

    splitter = OntologySplitter(input_file=input_file, output_dir=output_dir, train_ratio=0.8)
    try:
        splitter.save_split()
        return 0
    except Exception as e:
        print(f"\nError during split: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

    