"""Small offline tests for the read-only Project 6 rail holdout feasibility audit."""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_rail_holdouts as d


class StructuralTests(unittest.TestCase):
    def test_reachability_detects_transitive_leakage(self):
        edges = {("A", "B"), ("B", "C")}
        self.assertTrue(d.reaches(edges, "A", "C"))
        self.assertFalse(d.reaches(edges, "C", "A"))

    def test_split_requires_two_per_fold_and_uses_only_rail_classes(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / 'src').mkdir()
            (root / 'generated').mkdir()
            s = Graph()
            for i in range(421):
                # Original fixtures can include arbitrary labels/IRIs.
                s.add((URIRef(f"urn:old:{i}"), RDFS.label, URIRef("urn:label")))
            self.assertEqual(len(s), 421)
            src_path = root / 'src/module.ttl'
            # Force exact source byte prefix in augmented file.
            s.serialize(destination=src_path, format='turtle')
            new = Graph()
            for t in s: new.add(t)
            base = d.NS
            rail = sorted(d.EXPECTED)
            for name in rail:
                node = URIRef(base + name)
                new.add((node, RDF.type, OWL.Class))
                new.add((node, RDFS.subClassOf, URIRef('urn:parent:1')))
            # To fit real-stage byte-prefix+exact-count requirements, 6 property
            # declarations and 41 additional new-subject attributes are added.
            from rdflib import Literal
            props = []
            for i in range(4):
                p = URIRef(base+f'obj{i}')
                props.append(p)
                new.add((p, RDF.type, OWL.ObjectProperty))
            for i in range(2):
                p = URIRef(base+f'data{i}')
                props.append(p)
                new.add((p, RDF.type, OWL.DatatypeProperty))
            railrole = URIRef(base+'rail_passenger_role')
            new.add((railrole, RDFS.subClassOf, URIRef(d.PASSENGER)))
            for i in range(40):
                new.add((props[i % 6], RDFS.label, Literal(f'p{i}')))
            self.assertEqual(len(new), 421 + 16 + 6 + 1 + 40)
            self.assertEqual(len(new), 484)
            # Only one primary extra target, 9 total named subclass pairs.
            dest = root / 'generated/module_augmented.ttl'
            # N-Triples append is valid Turtle; serialize each new triple manually.
            delta = [t for t in new if t not in s]
            dest.write_bytes(src_path.read_bytes()+b'\n'+('\n'.join(f'{a.n3()} {b.n3()} {c.n3()} .' for a,b,c in delta)+'\n').encode())
            r = d.analyze(root, load_imports=False)
            self.assertEqual(r['summary']['folds'], 4)
            self.assertEqual(r['summary']['heldout_examples_per_fold'], [2,2,2,2])
            self.assertEqual(r['new_rail_classes'], 8)
            self.assertEqual(r['new_rail_subclass_relationships'], 9)
            self.assertTrue(r['files_unchanged'])
            self.assertFalse((root/'reports').exists())


if __name__ == '__main__':
    unittest.main()
