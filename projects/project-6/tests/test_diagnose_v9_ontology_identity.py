from pathlib import Path
import tempfile
import sys
import unittest

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_v9_ontology_identity import FILES, OUT, diagnose


class IdentityTest(unittest.TestCase):
    def make(self, root, graph1, graph2):
        for rel, g in zip(FILES, (graph1, graph2, Graph(), Graph(), Graph())):
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            g.serialize(destination=str(p), format="xml" if p.suffix == ".owl" else "turtle")

    def graphs(self, same=True):
        one = Graph()
        two = Graph()
        iri = URIRef("https://example.org/same-ontology")
        for g in (one, two):
            g.add((iri, RDF.type, OWL.Ontology))
        one.add((URIRef("https://example.org/A"), RDFS.subClassOf, URIRef("https://example.org/B")))
        two.add((URIRef("https://example.org/A"), RDFS.subClassOf, URIRef("https://example.org/B" if same else "https://example.org/C")))
        return one, two

    def test_equivalent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make(root, *self.graphs(same=True))
            data = diagnose(root)
            self.assertEqual(data["status"], "SAME_ONTOLOGY_IRI_RDF_EQUIVALENT")
            self.assertTrue(data["iao_comparison"]["rdf_isomorphic"])
            self.assertTrue((root / OUT).exists())

    def test_different(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make(root, *self.graphs(same=False))
            data = diagnose(root)
            self.assertEqual(data["status"], "SAME_ONTOLOGY_IRI_DIFFERENT_RDF_GRAPHS_REVIEW_REQUIRED")
            self.assertEqual(data["iao_comparison"]["left_only_named_subclass_count"], 1)
            self.assertFalse(data["iao_comparison"]["rdf_isomorphic"])


if __name__ == "__main__":
    unittest.main()
