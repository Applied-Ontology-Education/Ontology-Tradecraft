"""Unit tests for the isolated split; no Java or mOWL needed."""
from __future__ import annotations
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS, RDF, OWL

sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment_rail_mowl_v8 as m


class RailExperimentTests(unittest.TestCase):
    def fixture(self,root:Path):
        for name in ['src','generated','reports','imports','scripts']:
            (root/name).mkdir(parents=True,exist_ok=True)
        (root/'src/module.ttl').write_text('<urn:source:A> <urn:source:prop> <urn:source:B> .\n')
        # A handful of unrelated source/import subclass facts for repeatable tests.
        base={(f'urn:source:Class{i}',f'urn:source:Parent{i}') for i in range(10)}
        rail={pair for fold in m.FOLDS for pair in fold}
        # Additional retained parent to test anchored vs cold start.
        rail.add((m.BASE+'rail_passenger_role','http://purl.obolibrary.org/obo/BFO_0000023'))
        g=Graph()
        for c,p in base:
            g.add((URIRef(c),RDFS.subClassOf,URIRef(p)))
        for c,p in rail:
            g.add((URIRef(c),RDFS.subClassOf,URIRef(p)))
        # augmented graph is merged rail + source
        g.serialize(root/'generated/module_augmented.ttl',format='turtle')
        train=Graph()
        valid=Graph()
        for c,p in base:
            train.add((URIRef(c),RDFS.subClassOf,URIRef(p)))
        for c,p in rail:
            train.add((URIRef(c),RDF.type,OWL.Class))
            train.add((URIRef(p),RDF.type,OWL.Class))
        valid.add((URIRef(m.BASE+'rail_passenger_role'),RDFS.subClassOf,URIRef(m.AGENT+'passenger')))
        train.add((URIRef(m.BASE+'rail_passenger_role'),RDFS.subClassOf,URIRef('http://purl.obolibrary.org/obo/BFO_0000023')))
        train.serialize(root/'generated/mowl_rail_train.ttl',format='turtle')
        valid.serialize(root/'generated/mowl_rail_valid.ttl',format='turtle')
        report={'status':'PASS_SPLIT_READY_FOR_MOWL',
                'source_sha256':m.digest(root/'src/module.ttl'),
                'train_sha256':m.digest(root/'generated/mowl_rail_train.ttl'),
                'valid_sha256':m.digest(root/'generated/mowl_rail_valid.ttl')}
        (root/'reports/step5_split_report.json').write_text(json.dumps(report))
        return base,rail

    def test_split_isolation_leakage_and_determinism(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.fixture(root)
            snap=m.immutability_snapshot(root)
            inputs=m.load_split(root)
            fold_list,prep=m.prepare(root,inputs,snap)
            self.assertEqual(len(fold_list),4)
            self.assertEqual(sum(len(f['heldout']) for f in fold_list),8)
            self.assertEqual(prep['heldout_classes'],8)
            for fold in fold_list:
                self.assertFalse(set(fold['heldout'])&fold['training'])
                self.assertFalse(set(fold['heldout'])&m.closure(fold['training']))
                self.assertEqual(m.named_pairs(m.graph_load(fold['training_path'])),fold['training'])
            self.assertEqual(snap,m.immutability_snapshot(root))
            old=[m.digest(f['training_path']) for f in fold_list]
            newer,_=m.prepare(root,inputs,snap)
            self.assertEqual(old,[m.digest(f['training_path']) for f in newer])
            self.assertEqual(m.determine_support(fold_list[3]['training'],m.BASE+'rail_journey')['support_status'],'COLD_START_NO_TAXONOMY_RELATION')

    def test_hashed_inputs_protect_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.fixture(root)
            (root/'src/module.ttl').write_text('corrupted')
            with self.assertRaises(RuntimeError):
                m.load_split(root)

    def test_controls_are_not_true_ancestors(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.fixture(root)
            inp=m.load_split(root)
            ancestors=m.closure(inp['edges'])
            for fold in m.FOLDS:
                d=m.controls(inp['edges'],fold)
                for c,ps in d.items():
                    self.assertEqual(len(ps),3)
                    self.assertTrue(all((c,p) not in ancestors for p in ps))

if __name__=='__main__':
    unittest.main(verbosity=2)
