"""Offline synthetic tests: native mOWL is mocked ONLY in the integration test.
Run from project-6 root: python -m unittest discover -s tests -p 'test_experiment_rail_mowl_v9.py' -v
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import types
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
# Preload PyTorch optimizer internals before mocking the mOWL module.
# Prevents duplicate lazy Torch Library initialization in this test process.
import torch._dynamo
from rdflib import Graph,URIRef
from rdflib.namespace import RDFS,RDF,OWL

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import experiment_rail_mowl_v8 as v8
import experiment_rail_mowl_v9 as v9

REVIEW=Path(__file__).resolve().parent/'fixtures'/'approved_anchors_REFERENCE.json'


def fixture(root):
    for n in ('src','generated','reports','scripts','imports'):
        (root/n).mkdir(parents=True,exist_ok=True)
    (root/'src/module.ttl').write_text('<urn:test:a> <urn:test:p> <urn:test:b> .\n')
    (root/'scripts/run_all.py').write_text('# original main pipeline must remain unchanged\n')
    train=Graph(); valid=Graph();aug=Graph()
    pairs=set()
    for i in range(48):
        c='https://example.org/context/class'+str(i)
        p='https://example.org/context/super'+str(i%9)
        q='https://example.org/context/alt'+str(i%5)
        pairs.update(((c,p),(c,q)))
    for j,(c,p) in enumerate(sorted(pairs)):
        g=valid if j%5==0 else train
        g.add((URIRef(c),RDFS.subClassOf,URIRef(p)))
    rail={edge for fold in v8.FOLDS for edge in fold}
    rail.add((v8.BASE+'rail_passenger_role',v8.BFO+'BFO_0000023'))
    for c,p in rail:
        aug.add((URIRef(c),RDFS.subClassOf,URIRef(p)))
        train.add((URIRef(c),RDF.type,OWL.Class))
        train.add((URIRef(p),RDF.type,OWL.Class))
    train.add((URIRef(v8.BASE+'rail_passenger_role'),RDFS.subClassOf,URIRef(v8.BFO+'BFO_0000023')))
    for c,p in v8.APPROVED_SEVEN:
        train.add((URIRef(c),RDF.type,OWL.Class))
        train.add((URIRef(p),RDF.type,OWL.Class))
    train.serialize(root/'generated/mowl_rail_train.ttl',format='turtle')
    valid.serialize(root/'generated/mowl_rail_valid.ttl',format='turtle')
    aug.serialize(root/'generated/module_augmented.ttl',format='turtle')
    report={'status':'PASS_SPLIT_READY_FOR_MOWL',
        'source_sha256':v8.digest(root/'src/module.ttl'),
        'train_sha256':v8.digest(root/'generated/mowl_rail_train.ttl'),
        'valid_sha256':v8.digest(root/'generated/mowl_rail_valid.ttl')}
    (root/'reports/step5_split_report.json').write_text(json.dumps(report))
    review=root/'reports/experiments/rail_anchor_review/approved_anchors.json'
    review.parent.mkdir(parents=True,exist_ok=True)
    review.write_bytes(REVIEW.read_bytes())
    return review


class V9Tests(unittest.TestCase):
    def test_seven_anchor_review_and_separate_nested_folds(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);review=fixture(root)
            snap=v8.immutability_snapshot(root)
            inputs=v8.load_split(root)
            review,approved=v9.check_review(root,review,inputs)
            self.assertEqual(approved,v8.APPROVED_SEVEN)
            keep=root/'reports/experiments/rail_mowl_v8_reviewed/eight_class_summary.json'
            keep.parent.mkdir(parents=True,exist_ok=True)
            keep.write_text('V8 MUST STAY HERE')
            original_hash=v8.digest(keep)
            folds,report=v9.build_preparation(root,inputs,approved,snap,review,101,16)
            self.assertEqual(len(folds),4)
            self.assertEqual(report['approved_anchor_count'],7)
            self.assertEqual(report['outer_holdout_total'],8)
            self.assertEqual(v8.digest(keep),original_hash)
            self.assertEqual(v8.immutability_snapshot(root),snap)
            for f in folds:
                self.assertEqual(len(f['dev']),16)
                self.assertTrue(approved.issubset(f['inner_edges']))
                self.assertTrue(approved.issubset(f['outer_edges']))
                self.assertEqual(set(f['heldout'])&v8.closure(f['outer_edges']),set())
                self.assertEqual(set(f['dev'])&v8.closure(f['inner_edges']),set())
                self.assertFalse(any(c.startswith(v8.BASE) for c,p in f['dev']))
                self.assertTrue(all(s['training_outgoing'] for s in f['support'].values()))
                self.assertIn('v9',str(f['outer_path']))

    def test_fold_reachability_guard_rejects_leaked_targets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);review=fixture(root)
            p=root/'generated/mowl_rail_train.ttl'
            g=v8.graph_load(p)
            g.add((URIRef(v8.BFO+'BFO_0000004'),RDFS.subClassOf,URIRef(v8.BFO+'BFO_0000040')))
            g.serialize(p,format='turtle')
            r=root/'reports/step5_split_report.json'
            report=json.loads(r.read_text());report['train_sha256']=v8.digest(p);r.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'named superclass leakage'):
                inputs=v8.load_split(root)
                v9.check_review(root,review,inputs)

    def test_dev_controls_are_deterministic_and_not_known_named_ancestors(self):
        edges={(f'https://example.org/c{i}',f'https://example.org/p{i%5}') for i in range(28)}
        truth=[('https://example.org/c1','https://example.org/p1')]
        c1=v9.deterministic_controls(edges,truth,99)
        self.assertEqual(c1,v9.deterministic_controls(edges,truth,99))
        self.assertEqual(len(c1[truth[0]]),3)
        self.assertNotIn(truth[0][1],c1[truth[0]])

    def test_train_contrastive_controls_exclude_all_rail_children_and_dev(self):
        rail=v8.BASE+'rail_ticket'
        dev='https://example.org/context/dev'
        x='https://example.org/context/train'
        edges={(rail,'https://example.org/parents/a'),(dev,'https://example.org/parents/a'),
               (x,'https://example.org/parents/a'),
               ('https://example.org/context/other','https://example.org/parents/b'),
               ('https://example.org/context/other2','https://example.org/parents/c'),
               ('https://example.org/context/other3','https://example.org/parents/d')}
        idx={s:i for i,s in enumerate(sorted({w for e in edges for w in e}))}
        neg=v9.sample_negative_pairs(edges,idx,101,{dev})
        self.assertFalse(any(c in (dev,rail) for c,p,n in neg))
        self.assertFalse(any((c,n) in v8.closure(edges) for c,p,n in neg))

    def test_selection_requires_development_improvement_and_geometry(self):
        base={'auc':.91,'gap':.42}
        rows=[{'dev_auc':.92,'dev_gap':.43,'geometry_mean_absolute_shift':.20,
               'passes_dev_guards':False,'config':{'name':'collapsing'}},
              {'dev_auc':.94,'dev_gap':.44,'geometry_mean_absolute_shift':.05,
               'passes_dev_guards':True,'config':{'name':'preserving'}}]
        self.assertEqual(v9.select_config(rows,base)['config']['name'],'preserving')
        self.assertIsNone(v9.select_config(rows[:1],base))

    def test_projection_identity_and_geometry(self):
        v8.seed_everything(101)
        base=torch.randn((24,50))
        idx={f'https://test.org/C{i}':i for i in range(24)}
        training={(f'https://test.org/C{i}',f'https://test.org/C{i+1}') for i in range(0,16,2)}
        cfg=v9.configs(3)[0]
        projector,mat,loss=v9.fit_projection(base,idx,training,cfg,111,[],v9.geometry_pairs(idx,101,32))
        self.assertEqual(tuple(mat.shape),(24,50))
        self.assertTrue(torch.isfinite(mat).all())
        self.assertGreaterEqual(loss['last_training_loss'],0)
        geo=v9.geometry_report(base,mat,v9.geometry_pairs(idx,101,32))
        self.assertTrue(math_isfinite(geo['mean_absolute_shift']))

    def test_contrastive_loss_and_projection_selection_path(self):
        base=torch.randn((20,50),generator=torch.Generator().manual_seed(22))
        idx={f"https://example.org/test/C{i}":i for i in range(20)}
        edges={(f"https://example.org/test/C{i}",f"https://example.org/test/C{i+1}")
               for i in range(0,12,2)}
        neg=[(f"https://example.org/test/C{i}",f"https://example.org/test/C{i+1}",
              f"https://example.org/test/C{i+8}") for i in range(0,12,2)]
        cfg=v9.configs(2)[7]
        self.assertGreater(cfg['contrastive_weight'],0)
        model,after,loss=v9.fit_projection(base,idx,edges,cfg,55,neg,v9.geometry_pairs(idx,45,30))
        self.assertTrue(torch.isfinite(after).all())
        self.assertTrue(math_isfinite(loss['last_training_loss']))

    def test_real_selected_config_branch_with_fake_mowl(self):
        # Forces the selected branch solely for code coverage, not a model claim.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);review=fixture(root)
            inputs=v8.load_split(root);snapshot=v8.immutability_snapshot(root)
            review,approved=v9.check_review(root,review,inputs)
            folds,report=v9.build_preparation(root,inputs,approved,snapshot,review,101,8)
            def fake_fit_native(path,seed,epochs,dim):
                graph=v8.graph_load(path)
                nodes={str(s) for s in graph.subjects(RDF.type,OWL.Class)}
                nodes|={z for e in v8.named_pairs(graph) for z in e}
                index={iri:i for i,iri in enumerate(sorted(nodes))}
                mat=torch.randn((len(index),dim),generator=torch.Generator().manual_seed(seed))
                return index,mat,{'mode':'FAKE_SYNTHETIC_TEST_ONLY'}
            def first_candidate(rows,base):
                return rows[0]
            with patch.dict(sys.modules,{'mowl':types.SimpleNamespace(init_jvm=lambda _:None)}), \
                 patch.object(v8,'fit_native',side_effect=fake_fit_native), \
                 patch.object(v9,'select_config',side_effect=first_candidate):
                summary=v9.evaluate(root,folds,inputs,approved,snapshot,review,101,1,2,1)
            self.assertEqual(summary['projection_selected_on_development_folds'],4)
            self.assertEqual(summary['projection_fallback_to_native_folds'],0)
            self.assertTrue((root/'generated/experiments/rail_mowl_v9_reviewed/fold_2_selected_projection.pt').exists())
            self.assertEqual(snapshot,v8.immutability_snapshot(root))

    def test_end_to_end_with_mocked_native_training_no_java(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);review=fixture(root)
            inputs=v8.load_split(root);snap=v8.immutability_snapshot(root)
            review,approved=v9.check_review(root,review,inputs)
            folds,report=v9.build_preparation(root,inputs,approved,snap,review,101,8)
            def fake_fit_native(path,seed,epochs,dim):
                graph=v8.graph_load(path)
                nodes={str(s) for s in graph.subjects(RDF.type,OWL.Class)}
                nodes|={z for e in v8.named_pairs(graph) for z in e}
                index={iri:i for i,iri in enumerate(sorted(nodes))}
                g=torch.Generator().manual_seed(seed)
                matrix=torch.randn((len(index),dim),generator=g)
                return index,matrix,{'mode':'FAKE_SYNTHETIC_TEST_ONLY'}
            with patch.dict(sys.modules,{'mowl':types.SimpleNamespace(init_jvm=lambda _:None)}), \
                 patch.object(v8,'fit_native',side_effect=fake_fit_native):
                summary=v9.evaluate(root,folds,inputs,approved,snap,review,101,1,2,2)
            self.assertEqual(summary['folds_completed'],4)
            self.assertEqual(summary['outer_test_sample_size'],8)
            self.assertEqual(summary['configs_per_fold'],2)
            self.assertTrue(summary['all_8_rail_targets_unseen_during_model_selection'])
            self.assertEqual(v8.immutability_snapshot(root),snap)
            self.assertTrue((root/'reports/experiments/rail_mowl_v9_reviewed/all_pair_scores.csv').exists())
            self.assertFalse((root/'reports/experiments/rail_mowl_v8_reviewed/eight_class_summary.json').exists())


def math_isfinite(value):
    import math
    return math.isfinite(value)

if __name__=='__main__':unittest.main(verbosity=2)
