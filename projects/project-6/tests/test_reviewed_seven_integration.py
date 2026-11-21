"""Synthetic integration tests for seven human-approved anchors.

No user ontology data, Java, or online calls are needed.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from rdflib import Graph,URIRef
from rdflib.namespace import RDF,RDFS,OWL

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import experiment_rail_mowl_v8 as v8
import test_audit_rail_superclass_support as audit_tests

HERE=Path(__file__).resolve().parents[1]
APPROVAL=HERE/'reports/experiments/rail_anchor_review/approved_anchors.json'

class SevenAnchorTests(unittest.TestCase):
    def fixture(self,root):
        audit_tests.AnchorAuditTests().fixture(root)
        # The real project imports contain these BFO/IAO classes. Declare them
        # in the synthetic fixture to reproduce the inventory prerequisite.
        p=root/'generated/mowl_rail_train.ttl'
        g=v8.graph_load(p)
        for _,parent in v8.APPROVED_SEVEN:
            g.add((URIRef(parent),RDF.type,OWL.Class))
        g.serialize(p,format='turtle')
        split=root/'reports/step5_split_report.json'
        report=json.loads(split.read_text())
        report['train_sha256']=v8.digest(p)
        split.write_text(json.dumps(report))
        return v8.load_split(root)

    def test_integrates_all_seven_into_every_fold_but_not_the_holdouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inputs=self.fixture(root)
            snap=v8.immutability_snapshot(root)
            baseline,_=v8.prepare(root,inputs,snap)
            original_hashes={f['fold']:f['train_sha256'] for f in baseline}
            folds,prep,extended=v8.prepare_approved(root,inputs,snap,APPROVAL)
            self.assertEqual(prep['approved_anchor_count'],7)
            self.assertFalse(prep['full_owl_semantic_entailment_checked'])
            self.assertFalse(prep['named_subclass_entailment_leakage_found'])
            self.assertEqual(extended['reviewed_anchors'],v8.APPROVED_SEVEN)
            self.assertEqual(len(folds),4)
            for f in folds:
                observed=v8.named_pairs(v8.graph_load(f['training_path']))
                self.assertEqual(observed,f['training'])
                self.assertTrue(v8.APPROVED_SEVEN.issubset(observed))
                self.assertEqual(len(observed),len(inputs['edges'])-2+7)
                self.assertEqual(set(f['heldout']) & observed,set())
                self.assertEqual(set(f['heldout']) & v8.closure(observed),set())
                self.assertTrue(all(row['training_outgoing']>=1 for row in f['support'].values()))
                self.assertTrue(all(row['support_status']=='RETAINED_PARENT_SUPPORT' for row in f['support'].values()))
                self.assertEqual(f['train_sha256'],v8.digest(f['training_path']))
            for f in baseline:
                self.assertEqual(v8.digest(f['training_path']),original_hashes[f['fold']])
            self.assertEqual(snap,v8.immutability_snapshot(root))
            # Known approved positive for passenger_train is NOT a control
            # for its held-out Train relation.
            c=v8.controls(extended['edges'],v8.FOLDS[3])
            self.assertNotIn(v8.BFO+'BFO_0000040',c[v8.BASE+'passenger_train'])

    def test_reviewed_anchors_not_accepted_if_one_is_missing_or_substituted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inputs=self.fixture(root)
            snap=v8.immutability_snapshot(root)
            payload=json.loads(APPROVAL.read_text())
            for changed in [payload['approved_anchors'][:-1],
                            [dict(item) if i else {**item,'parent':v8.BFO+'BFO_0000015'}
                             for i,item in enumerate(payload['approved_anchors'])]]:
                data={**payload,'approved_anchors':changed}
                f=root/'wrong.json';f.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,'Exactly the seven'):
                    v8.prepare_approved(root,inputs,snap,f)
            self.assertEqual(snap,v8.immutability_snapshot(root))

    def test_rejects_transitively_leaking_ancestor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root)
            path=root/'generated/mowl_rail_train.ttl'
            g=v8.graph_load(path)
            # In the real BFO taxonomy IndependentContinuant is broader
            # than MaterialEntity, NOT its subclass. Reverse it for a leak.
            g.add((URIRef(v8.BFO+'BFO_0000004'),RDFS.subClassOf,
                   URIRef(v8.BFO+'BFO_0000040')))
            g.serialize(path,format='turtle')
            split=root/'reports/step5_split_report.json'
            d=json.loads(split.read_text());d['train_sha256']=v8.digest(path)
            split.write_text(json.dumps(d))
            inputs=v8.load_split(root)
            with self.assertRaisesRegex(ValueError,'create named superclass leakage'):
                v8.prepare_approved(root,inputs,v8.immutability_snapshot(root),APPROVAL)

if __name__=='__main__':
    unittest.main(verbosity=2)
