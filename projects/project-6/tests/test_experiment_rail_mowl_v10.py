"""Synthetic regression tests for isolated v10 cosine experiment.

From project-6 root:
 python -m unittest discover -s tests -p 'test_experiment_rail_mowl_v10.py' -v
Native mOWL is mocked only in test_full_synthetic; real model is not tested.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch._dynamo  # Load optimizer internals before mOWL test patching.
from rdflib import Graph,URIRef
from rdflib.namespace import RDF,OWL,RDFS

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import experiment_rail_mowl_v8 as v8
import experiment_rail_mowl_v9 as v9
import experiment_rail_mowl_v10 as v10

REVIEW=Path(__file__).resolve().parent/'fixtures/approved_anchors_REFERENCE.json'


def fixture(root):
    for n in ('src','generated','reports','scripts','imports'):
        (root/n).mkdir(parents=True,exist_ok=True)
    (root/'src/module.ttl').write_text('<urn:test:subject> <urn:test:pred> <urn:test:object> .\n',encoding='utf-8')
    (root/'scripts/run_all.py').write_text('# UNCHANGED\n',encoding='utf-8')
    # For local offline tests copy the upstream module files by contents, no source edits.
    for n,mod in (('experiment_rail_mowl_v8.py',v8),('experiment_rail_mowl_v9.py',v9)):
        (root/'scripts'/n).write_bytes(Path(mod.__file__).read_bytes())
    pairs=set()
    for i in range(50):
        c=f'https://example.org/context/class{i}'
        p=f'https://example.org/context/super{i%8}'
        q=f'https://example.org/context/alt{i%7}'
        pairs.update(((c,p),(c,q)))
    training=Graph();validation=Graph();aug=Graph()
    for j,(c,p) in enumerate(sorted(pairs)):
        (validation if j%5==0 else training).add((URIRef(c),RDFS.subClassOf,URIRef(p)))
    expected={pair for fold in v8.FOLDS for pair in fold}
    expected.add((v8.BASE+'rail_passenger_role',v8.BFO+'BFO_0000023'))
    for c,p in expected:
        aug.add((URIRef(c),RDFS.subClassOf,URIRef(p)))
        training.add((URIRef(c),RDF.type,OWL.Class))
        training.add((URIRef(p),RDF.type,OWL.Class))
    training.add((URIRef(v8.BASE+'rail_passenger_role'),RDFS.subClassOf,
                  URIRef(v8.BFO+'BFO_0000023')))
    for c,p in v8.APPROVED_SEVEN:
        training.add((URIRef(c),RDF.type,OWL.Class))
        training.add((URIRef(p),RDF.type,OWL.Class))
    training.serialize(root/'generated/mowl_rail_train.ttl',format='turtle')
    validation.serialize(root/'generated/mowl_rail_valid.ttl',format='turtle')
    aug.serialize(root/'generated/module_augmented.ttl',format='turtle')
    report={'status':'PASS_SPLIT_READY_FOR_MOWL',
        'source_sha256':v8.digest(root/'src/module.ttl'),
        'train_sha256':v8.digest(root/'generated/mowl_rail_train.ttl'),
        'valid_sha256':v8.digest(root/'generated/mowl_rail_valid.ttl')}
    (root/'reports/step5_split_report.json').write_text(json.dumps(report),encoding='utf-8')
    review=root/'reports/experiments/rail_anchor_review/approved_anchors.json'
    review.parent.mkdir(parents=True,exist_ok=True)
    review.write_bytes(REVIEW.read_bytes())
    snap=v8.immutability_snapshot(root)
    frozen_path=root/v10.FROZEN_V9
    frozen_path.parent.mkdir(parents=True,exist_ok=True)
    summary={'version':v9.VERSION,'folds_completed':4,'projection_fallback_to_native_folds':4,
        'outer_test_sample_size':8,'all_8_rail_targets_unseen_during_model_selection':True,
        'native_8_class_mean':.6037136092782021,'native_auc':.953125,
        'native_gap':.4515440862160176,'native_top1_of_four':7,'input_sha256':snap}
    frozen_path.write_text(json.dumps(summary),encoding='utf-8')
    for rel in v10.REFERENCE_PATHS:
        p=root/rel
        if p.is_file(): continue
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('FROZEN v9 PLACEHOLDER FOR SYNTHETIC TEST\n',encoding='utf-8')
    return review


class V10Tests(unittest.TestCase):
    def test_prepare_no_v9_mutation_and_four_folds(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=fixture(root)
            snap=v8.immutability_snapshot(root)
            ref=v10.check_frozen_v9(root,snap)
            hashes=ref['reference_files_sha256'].copy()
            inputs=v8.load_split(root)
            review,anchors=v9.check_review(root,r,inputs)
            folds,prep=v10.prepare(root,inputs,anchors,snap,review,101,16,ref)
            self.assertEqual(len(folds),4)
            self.assertEqual(prep['outer_holdout_total'],8)
            self.assertEqual(prep['version'],v10.VERSION)
            self.assertEqual(hashes,ref['reference_files_sha256'])
            self.assertEqual(v9.TAG,'rail_mowl_v9_reviewed')
            self.assertEqual(snap,v8.immutability_snapshot(root))
            for f in folds:
                self.assertIn(v10.TAG,str(f['outer_path']))
                self.assertEqual(len(f['dev']),16)
                self.assertEqual(len(f['inner_edges']),len(f['outer_edges'])-16)
                self.assertTrue(anchors.issubset(f['inner_edges']))
                self.assertFalse(set(f['heldout']) & v8.closure(f['outer_edges']))
                self.assertFalse(set(f['dev']) & v8.closure(f['inner_edges']))
            v10.check_reference_unchanged(root,ref)

    def test_freeze_validator_rejects_metrics_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            snap=v8.immutability_snapshot(root)
            v10.check_frozen_v9(root,snap)
            p=root/v10.FROZEN_V9
            j=json.loads(p.read_text());j['native_auc']=.1;p.write_text(json.dumps(j))
            with self.assertRaisesRegex(ValueError,'unexpectedly differ'):
                v10.check_frozen_v9(root,snap)

    def test_freeze_validator_rejects_unexpected_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            ref=v10.check_frozen_v9(root,v8.immutability_snapshot(root))
            path=root/'reports/experiments/rail_mowl_v9_reviewed/fold_1_metrics.json'
            path.write_text('CHANGED\n')
            with self.assertRaisesRegex(RuntimeError,'modified'):
                v10.check_reference_unchanged(root,ref)

    def test_hard_negatives_not_rail_or_development_and_not_reachable(self):
        rail=v8.BASE+'rail_ticket'
        dev='https://test.org/dev'
        pairs={(rail,'https://test.org/P0'),(dev,'https://test.org/P0')}
        for j in range(6):
            pairs.add((f'https://test.org/c{j}',f'https://test.org/P{j%3}'))
            pairs.add((f'https://test.org/c{j}',f'https://test.org/Q{j%2}'))
        idx={u:i for i,u in enumerate(sorted({x for a,b in pairs for x in (a,b)}))}
        emb=torch.randn((len(idx),50),generator=torch.Generator().manual_seed(22))
        negatives=v10.hard_negative_pairs(pairs,idx,emb,101,{dev})
        self.assertTrue(negatives)
        closure=v8.closure(pairs)
        for c,p,q in negatives:
            self.assertNotIn(c,(rail,dev))
            self.assertNotIn((c,q),closure)
            self.assertNotEqual(p,q)

    def test_hard_negatives_choose_max_cosine_parent(self):
        edges={(f'https://t.org/c{i}',f'https://t.org/p{i}') for i in range(4)}
        idx={x:i for i,x in enumerate(sorted({x for a,b in edges for x in (a,b)}))}
        emb=torch.randn((len(idx),10),generator=torch.Generator().manual_seed(101))
        negatives=v10.hard_negative_pairs(edges,idx,emb,1,per_positive=1)
        import torch.nn.functional as F
        x=F.normalize(emb,dim=1)
        reach=v8.closure(edges)
        for c,p,alt in negatives:
            choices=[q for a,q in edges if q!=p and q!=c and (c,q) not in reach]
            self.assertAlmostEqual(float(x[idx[c]] @ x[idx[alt]]),
                max(float(x[idx[c]] @ x[idx[q]]) for q in choices),places=5)

    def test_cosine_inflation_must_preserve_discrimination_and_geometry(self):
        baseline={'auc':.63,'gap':.08,'paired_win_rate':.62,
            'top1_of_four':18,'positive_mean':.43}
        improved={'auc':.635,'gap':.081,'paired_win_rate':.62,
            'top1_of_four':18,'positive_mean':.49,
            'geometry':{'mean_absolute_shift':.03}}
        self.assertTrue(v10.selection_guard(improved,baseline))
        collapse={**improved,'positive_mean':.99,'auc':.40,'gap':.005}
        self.assertFalse(v10.selection_guard(collapse,baseline))
        overshift={**improved,'geometry':{'mean_absolute_shift':.30}}
        self.assertFalse(v10.selection_guard(overshift,baseline))
        no_gain={**improved,'positive_mean':.435}
        self.assertFalse(v10.selection_guard(no_gain,baseline))

    def test_choose_development_only_or_native_fallback(self):
        baseline={'auc':.62,'gap':.06,'positive_mean':.44}
        poor=[{'config':{'name':'A'},'passes_dev_guards':False,'dev_auc':.98,
             'dev_gap':.9,'dev_positive_mean':.99,'geometry_mean_absolute_shift':.5}]
        self.assertIsNone(v10.choose(poor,baseline))
        good=[{'config':{'name':'B'},'passes_dev_guards':True,'dev_auc':.64,
             'dev_gap':.07,'dev_positive_mean':.52,'geometry_mean_absolute_shift':.02}]
        self.assertEqual(v10.choose(poor+good,baseline)['config']['name'],'B')

    def test_config_count_and_nontrivial_contrastive(self):
        cs=v10.configs(3)
        self.assertEqual(len(cs),8)
        self.assertEqual({a['epochs'] for a in cs},{3})
        self.assertTrue(all(a['contrastive_weight']>=1 for a in cs))
        self.assertTrue(all(a['positive_weight']>0 for a in cs))

    def test_full_synthetic_with_fake_native_mowl(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);review_file=fixture(root)
            snap=v8.immutability_snapshot(root)
            reference=v10.check_frozen_v9(root,snap)
            inputs=v8.load_split(root)
            review,anchors=v9.check_review(root,review_file,inputs)
            folds,_=v10.prepare(root,inputs,anchors,snap,review,101,8,reference)
            def fake_native(path,seed,epochs,dim):
                g=v8.graph_load(path)
                ids={str(s) for s in g.subjects(RDF.type,OWL.Class)}
                ids|={n for e in v8.named_pairs(g) for n in e}
                idx={s:j for j,s in enumerate(sorted(ids))}
                mat=torch.randn((len(idx),dim),generator=torch.Generator().manual_seed(seed))
                return idx,mat,{'synthetic':'NOT_REAL_MOWL'}
            with patch.dict(sys.modules,{'mowl':types.SimpleNamespace(init_jvm=lambda _:None)}), \
                 patch.object(v8,'fit_native',side_effect=fake_native):
                summary=v10.evaluate(root,folds,inputs,anchors,snap,review,reference,
                                      101,1,2,2)
            self.assertEqual(summary['folds_completed'],4)
            self.assertEqual(summary['rail_holdouts'],8)
            self.assertEqual(summary['controls'],24)
            self.assertEqual(summary['configuration_count'],2)
            self.assertFalse(summary['target_edges_used_for_selection'])
            self.assertEqual(v8.immutability_snapshot(root),snap)
            self.assertEqual(v9.TAG,'rail_mowl_v9_reviewed')
            v10.check_reference_unchanged(root,reference)
            self.assertTrue((root/'reports/experiments/rail_mowl_v10_cosine/all_pair_scores.csv').exists())
            self.assertFalse((root/'reports/experiments/rail_mowl_v9_reviewed/fold_1_vectors.npz').exists())


    def test_full_synthetic_forced_selection_checkpoint_path(self):
        # Force the selected branch to test code paths, not claim scientific fitness.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);review_file=fixture(root)
            snap=v8.immutability_snapshot(root)
            reference=v10.check_frozen_v9(root,snap)
            inputs=v8.load_split(root)
            review,anchors=v9.check_review(root,review_file,inputs)
            folds,_=v10.prepare(root,inputs,anchors,snap,review,101,8,reference)
            def fake_native(path,seed,epochs,dim):
                g=v8.graph_load(path)
                ids={str(s) for s in g.subjects(RDF.type,OWL.Class)}
                ids|={n for e in v8.named_pairs(g) for n in e}
                idx={s:j for j,s in enumerate(sorted(ids))}
                mat=torch.randn((len(idx),dim),generator=torch.Generator().manual_seed(seed))
                return idx,mat,{'synthetic':'NOT_REAL_MOWL'}
            with patch.dict(sys.modules,{'mowl':types.SimpleNamespace(init_jvm=lambda _:None)}), \
                 patch.object(v8,'fit_native',side_effect=fake_native), \
                 patch.object(v10,'choose',side_effect=lambda rows,baseline:rows[0]):
                summary=v10.evaluate(root,folds,inputs,anchors,snap,review,reference,
                                      101,1,2,1)
            self.assertEqual(summary['selected_projection_folds'],4)
            self.assertEqual(summary['native_fallback_folds'],0)
            for i in range(1,5):
                self.assertTrue((root/f'generated/experiments/{v10.TAG}/fold_{i}_selected_projection.pt').exists())
            v10.check_reference_unchanged(root,reference)


if __name__=='__main__':
    unittest.main(verbosity=2)
