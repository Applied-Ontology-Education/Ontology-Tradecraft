"""Project 6 v10: development-guarded discriminative COSINE experiment.

Run from projects/project-6:
    python -u scripts/experiment_rail_mowl_v10.py --prepare-only
    python -u scripts/experiment_rail_mowl_v10.py --native-epochs 3 --projection-epochs 3 --configs 2
    python -u scripts/experiment_rail_mowl_v10.py

Requires the EXISTING v8 and v9 scripts and reviewed approved_anchors.json.
Uses identical outer and inner folds and native model seeds as v9 for a
matched-design reference; DOES NOT edit v9 scripts, data, or reports.

Learn on named GCI0 source/import + approved broader anchors only. Tune
projection hyperparameters on development assertions hidden from the inner
native model AND projector. Eight outer rail relationships are unseen during
selection. Hard contrastive alternatives are unasserted, NOT OWL false axioms.
A previously successful HermiT named-only audit does not resolve full-OWL
import closure. This experiment does not assert complete OWL semantic validity.

Writes only generated/experiments/rail_mowl_v10_cosine and
reports/experiments/rail_mowl_v10_cosine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import experiment_rail_mowl_v8 as v8
import experiment_rail_mowl_v9 as v9

TAG = 'rail_mowl_v10_cosine'
VERSION = 'project6-v10-development-guarded-hard-contrastive-cosine-v1'
FROZEN_V9 = Path('reports/experiments/rail_mowl_v9_reviewed/eight_class_summary.json')
REFERENCE_PATHS = [
    'scripts/experiment_rail_mowl_v8.py',
    'scripts/experiment_rail_mowl_v9.py',
    'reports/experiments/rail_mowl_v9_reviewed/eight_class_summary.json',
    'reports/experiments/rail_mowl_v9_reviewed/preparation.json',
    'reports/experiments/rail_mowl_v9_reviewed/all_pair_scores.csv',
] + [f'reports/experiments/rail_mowl_v9_reviewed/fold_{i}_metrics.json' for i in range(1,5)]


def log(message):
    print('[Rail mOWL v10] '+str(message),flush=True)


def check_frozen_v9(root, snapshot):
    """Require a completed v9 reference, record immutable input/output digests."""
    path = root / FROZEN_V9
    if not path.is_file():
        raise FileNotFoundError('Frozen v9 summary missing; do not reconstruct it by running v9 again: '+str(path))
    raw = json.loads(path.read_text(encoding='utf-8'))
    if (raw.get('version') != v9.VERSION or raw.get('folds_completed') != 4 or
        raw.get('projection_fallback_to_native_folds') != 4 or
        raw.get('outer_test_sample_size') != 8 or
        raw.get('all_8_rail_targets_unseen_during_model_selection') is not True):
        raise ValueError('v9 report does not match completed native-only 4-fold baseline')
    if (abs(float(raw.get('native_8_class_mean', 0)) - .6037136092782021) > 1e-6 or
        abs(float(raw.get('native_auc', 0)) - .953125) > 1e-6):
        raise ValueError('v9 baseline metrics unexpectedly differ from frozen reported results')
    for rel, digest in raw.get('input_sha256',{}).items():
        if rel not in snapshot or snapshot[rel] != digest:
            raise RuntimeError(f'Frozen v9 input provenance changed: {rel}')
    paths = {}
    missing=[]
    for rel in REFERENCE_PATHS:
        p=root/rel
        if p.is_file(): paths[rel]=v8.digest(p)
        else: missing.append(rel)
    if missing:
        raise FileNotFoundError('Missing frozen v9 reference artifacts:\n'+'\n'.join(missing))
    return {'reference':'FULL_V9_NATIVE_ONLY','v9_metric_summary':{
        'native_8_class_mean':raw['native_8_class_mean'],
        'native_auc':raw['native_auc'],'native_gap':raw['native_gap'],
        'native_top1_of_four':raw['native_top1_of_four']},
        'reference_files_sha256':paths,
        'v9_frozen_archive_present':(root/'reports/experiments/v9_frozen_2026_11_19/v9_frozen_snapshot.zip').is_file()}


def check_reference_unchanged(root, reference):
    for rel, before in reference['reference_files_sha256'].items():
        if v8.digest(root/rel)!=before:
            raise RuntimeError('Frozen v9 reference file modified during v10 run: '+rel)


def prepare(root,inputs,anchors,snapshot,review,seed,dev_size,reference):
    # The v9 fold writer reads TAG only for its output paths. Temporarily
    # change the in-memory module attribute, then restore it. Never change
    # the source file or write to a v9 path.
    before=v9.TAG
    try:
        v9.TAG=TAG
        folds,manifest=v9.build_preparation(root,inputs,anchors,snapshot,review,seed,dev_size)
    finally:
        v9.TAG=before
    manifest['version']=VERSION
    manifest['mode']='V10_HARD_CONTRASTIVE_DEVELOPMENT_ONLY_COSINE'
    manifest['frozen_v9_file_hashes_recorded']=True
    manifest['reasoner_status']='NAMED_ONLY_PASSED_FULL_IMPORT_CLOSURE_NOT_COMPLETED'
    v8.save_json(root/'reports/experiments'/TAG/'preparation.json',manifest)
    v8.save_json(root/'reports/experiments'/TAG/'v9_reference_manifest.json',reference)
    check_reference_unchanged(root,reference)
    return folds,manifest


def configs(epochs):
    """Frozen grid before seeing rail scores; emphasis on ranking + cosine."""
    presets = [
        # name, scale, identity, contrastive, geometry, margin, temperature, positive
        ('hard_rank_light', .10, 2.0, 1.0,.20,.15,.15,.75),
        ('hard_rank_medium',.10, 2.0, 2.0,.20,.20,.15,.75),
        ('hard_rank_strong',.10, 3.0, 3.0,.30,.20,.15,1.00),
        ('gentle_hard_rank',.05,3.0, 2.0,.30,.15,.15,1.00),
        ('cosine_and_rank',.10,1.5, 1.5,.20,.15,.15,1.50),
        ('cosine_conservative',.05,2.0,1.0,.25,.10,.20,1.50),
        ('high_identity_rank',.10,4.0,3.0,.35,.15,.15,1.00),
        ('max_guarded_rank',.05,4.0,4.0,.40,.20,.10,1.00),
    ]
    return [dict(name=name,epochs=epochs,hidden_size=32,learning_rate=.0025,
            residual_scale=scale,identity_weight=identity,
            contrastive_weight=contrastive,geometry_weight=geometry,
            margin=margin,temperature=temp,weight_decay=1e-5,
            positive_weight=positive,residual_penalty_weight=.001)
        for name,scale,identity,contrastive,geometry,margin,temp,positive in presets]


def hard_negative_pairs(training,index,base,seed,exclude_children=(),per_positive=2):
    """Training-only, high-cosine named-unentailed alternatives.

    No outer rail child, no inner-dev child, no withheld rail pair can enter
    gradient examples. Deterministic ordering on tied cosine values.
    These are never declared false axioms under OWL open world semantics.
    """
    import torch.nn.functional as F
    excluded=set(exclude_children)|{c for fold in v8.FOLDS for c,_ in fold}
    candidate_parents=sorted({p for c,p in training if p in index})
    ancestors=v8.closure(training)
    normal=F.normalize(base.detach(),dim=1)
    out=[]
    for c,p in sorted(training):
        if c in excluded or c not in index or p not in index:
            continue
        options=[q for q in candidate_parents if q!=c and q!=p and (c,q) not in ancestors]
        # Sorting by cosine DESC, stable lexicographic IRI on ties.
        options.sort(key=lambda q:(-float((normal[index[c]]*normal[index[q]]).sum()),q))
        for q in options[:per_positive]: out.append((c,p,q))
    if not out:
        raise ValueError('No admissible training-only hard comparison parents')
    return out


def selection_guard(candidate,baseline):
    """RELATIVE development gates prevent cosine inflation from winning."""
    return (candidate['auc'] >= baseline['auc']-.01
            and candidate['gap'] >= baseline['gap']-.01
            and candidate['paired_win_rate'] >= baseline['paired_win_rate']-.02
            and candidate['top1_of_four'] >= baseline['top1_of_four']-1
            and candidate['geometry']['mean_absolute_shift'] <= .08
            and candidate['positive_mean'] >= baseline['positive_mean']+.02)


def choose(candidate_rows,baseline):
    eligible=[r for r in candidate_rows if r['passes_dev_guards']]
    if not eligible: return None
    # Selection uses ONLY source/import development evaluation; no rail scores.
    return max(eligible,key=lambda r:(round(r['dev_auc']-baseline['auc'],8),
                       round(r['dev_gap']-baseline['gap'],8),
                       round(r['dev_positive_mean']-baseline['positive_mean'],8),
                       -r['geometry_mean_absolute_shift'],r['config']['name']))


def evaluate(root,folds,inputs,anchors,snapshot,review,reference,seed,native_epochs,proj_epochs,cfg_count):
    import torch
    import mowl
    mowl.init_jvm('2g')
    report_dir=root/'reports/experiments'/TAG
    graph_dir=root/'generated/experiments'/TAG
    cfgs=configs(proj_epochs)[:cfg_count]
    all_rows=[]
    summaries=[]
    for f in folds:
        i=f['fold']
        for kind in ('outer','inner'):
            if v8.digest(f[kind+'_path'])!=f[kind+'_sha256']:
                raise RuntimeError('v10 training fold graph changed after preparation')
        log(f'Fold {i}/4: INNER native ELEmbeddings ({native_epochs} epochs)')
        inner_idx,inner_emb,_=v8.fit_native(f['inner_path'],seed+i*11,native_epochs,v8.EMBED_DIM)
        dev_controls=v9.deterministic_controls(f['outer_edges'],f['dev'],seed+i*41)
        required={x for c,p in f['dev'] for x in (c,p)}
        required|={x for xs in dev_controls.values() for x in xs}
        if required-set(inner_idx): raise RuntimeError('Inner model index omits development class')
        native_dev=v9.stats(inner_emb,inner_idx,f['dev'],dev_controls)
        dev_excluded={c for c,p in f['dev']}
        neg=hard_negative_pairs(f['inner_edges'],inner_idx,inner_emb,seed+i*33,dev_excluded)
        geo=v9.geometry_pairs(inner_idx,seed+i*202)
        candidates=[]
        log(f'Fold {i}/4: {len(cfgs)} hard-negative cosine projections (DEV only)')
        for j,cfg in enumerate(cfgs):
            _,after,loss=v9.fit_projection(inner_emb,inner_idx,f['inner_edges'],cfg,
                                          seed+i*100+j,neg,geo)
            dev=v9.stats(after,inner_idx,f['dev'],dev_controls)
            geometry=v9.geometry_report(inner_emb,after,geo)
            passes=selection_guard({**dev,'geometry':geometry},native_dev)
            candidates.append({'config':cfg,'dev_auc':dev['auc'],'dev_gap':dev['gap'],
                'dev_positive_mean':dev['positive_mean'],
                'dev_control_mean':dev['control_mean'],
                'dev_paired_win_rate':dev['paired_win_rate'],
                'dev_top1_of_four':dev['top1_of_four'],
                'geometry_mean_absolute_shift':geometry['mean_absolute_shift'],
                'passes_dev_guards':passes,'training_hard_negative_triples':len(neg),
                'loss':loss})
        selected=choose(candidates,native_dev)
        chosen=selected['config'] if selected else None
        dev_report={'fold':i,'development_target_count':len(f['dev']),
            'native_dev':{k:v for k,v in native_dev.items() if k!='rows'},
            'configurations':candidates,'selected_config':chosen,
            'selection_rule':'development relative AUC/gap/ranking/geometry and +0.02 positive cosine; rail unseen',
            'hard_negative_disclaimer':'named-unentailed, not OWL proven false',
            'fallback_native':not bool(chosen),'dev_targets_excluded_from_gradients':True}
        v8.save_json(report_dir/f'fold_{i}_development.json',dev_report)
        log(f'Fold {i}/4: development selected '+(chosen['name'] if chosen else 'NATIVE_ONLY'))
        log(f'Fold {i}/4: OUTER native ELEmbeddings ({native_epochs} epochs)')
        idx,base,versions=v8.fit_native(f['outer_path'],seed+i*11+1,native_epochs,v8.EMBED_DIM)
        holdout=f['heldout']
        ctrl=v8.controls(set(inputs['edges'])|anchors,tuple(holdout))
        cmap={(c,p):ctrl[c] for c,p in holdout}
        required={x for c,p in holdout for x in (c,p)}
        required|={p for parents in ctrl.values() for p in parents}
        if required-set(idx):raise RuntimeError('Outer model index omits rail target/control')
        native_outer=v9.stats(base,idx,holdout,cmap)
        geo_outer=v9.geometry_pairs(idx,seed+i*808)
        if chosen:
            neg_outer=hard_negative_pairs(f['outer_edges'],idx,base,seed+i*52)
            model,after,fit_info=v9.fit_projection(base,idx,f['outer_edges'],chosen,
                seed+i*100+1000,neg_outer,geo_outer)
            torch.save({'state_dict':model.state_dict(),'config':chosen,'fold':i,
                'outer_training_sha256':f['outer_sha256'],
                'selection_source':'inner_source_import_only'},
                graph_dir/f'fold_{i}_selected_projection.pt')
        else:
            # Never leave behind a misleading checkpoint from an earlier run
            # of this same v10 output directory when fallback is selected.
            stale=graph_dir/f'fold_{i}_selected_projection.pt'
            if stale.is_file():
                stale.unlink()
            after=base.clone()
            fit_info=None
        v10_outer=v9.stats(after,idx,holdout,cmap)
        geometry_outer=v9.geometry_report(base,after,geo_outer)
        for n,a in zip(native_outer['rows'],v10_outer['rows']):
            all_rows.append({'fold':i,'record_type':'HELDOUT','child':n['child'],
                'expected_parent':n['expected_parent'],'native_cosine':n['cosine'],
                'v10_cosine':a['cosine'],'native_control_mean':n['control_mean'],
                'v10_control_mean':a['control_mean'],'native_rank':n['rank_among_four'],
                'v10_rank':a['rank_among_four'],'selected_config':chosen['name'] if chosen else 'NATIVE_ONLY'})
            for (parent,ns),(parent2,ps) in zip(n['controls'],a['controls']):
                if parent!=parent2:raise RuntimeError('Control IRI mismatch')
                all_rows.append({'fold':i,'record_type':'UNASSERTED_PARENT_CONTROL',
                    'child':n['child'],'control_parent':parent,
                    'native_cosine':ns,'v10_cosine':ps,
                    'selected_config':chosen['name'] if chosen else 'NATIVE_ONLY'})
        fold_summary={'fold':i,'selection':chosen['name'] if chosen else 'NATIVE_ONLY',
            'native':native_outer,'v10':v10_outer,'geometry':geometry_outer,
            'selected_config':chosen,'fit_details':fit_info,
            'input_training_sha256':f['outer_sha256'],
            'native_library_versions':versions,'rail_holdouts_unseen_during_tuning':True}
        v8.save_json(report_dir/f'fold_{i}_metrics.json',fold_summary)
        summaries.append(fold_summary)
        log(f"Fold {i}: native {native_outer['positive_mean']:.4f}; "
            f"v10 {v10_outer['positive_mean']:.4f}; v10 gap {v10_outer['gap']:.4f}; "
            f"v10 AUC {v10_outer['auc']:.3f}")
    cols=['fold','record_type','child','expected_parent','control_parent','native_cosine',
        'v10_cosine','native_control_mean','v10_control_mean','native_rank','v10_rank','selected_config']
    v8.save_csv(report_dir/'all_pair_scores.csv',all_rows,cols)
    pos=[r for r in all_rows if r['record_type']=='HELDOUT']
    ctrl=[r for r in all_rows if r['record_type']=='UNASSERTED_PARENT_CONTROL']
    native_pos=[r['native_cosine'] for r in pos]
    v10_pos=[r['v10_cosine'] for r in pos]
    native_ctrl=[r['native_cosine'] for r in ctrl]
    v10_ctrl=[r['v10_cosine'] for r in ctrl]
    native_gap=v8.mean(native_pos)-v8.mean(native_ctrl)
    v10_gap=v8.mean(v10_pos)-v8.mean(v10_ctrl)
    native_auc=v8.rank_auc(native_pos,native_ctrl)
    v10_auc=v8.rank_auc(v10_pos,v10_ctrl)
    mean_gate=v8.mean(v10_pos)>=v8.TARGET
    v10_top1=sum(r['v10_rank']==1 for r in pos)
    native_top1=sum(r['native_rank']==1 for r in pos)
    # The outer diagnostic pass rule is relative to a matched-seed native run.
    # It is NOT used to tune or select a model.
    control_gate=(v10_auc>=max(.70,native_auc-.03)
                  and v10_gap>=max(.15,native_gap-.05)
                  and v10_top1>=native_top1-1)
    summary={'version':VERSION,
        'status':'TARGET_AND_CONTROLS_MET_DIAGNOSTIC_ONLY' if mean_gate and control_gate
              else 'TARGET_OR_CONTROLS_NOT_MET',
        'mode':'NESTED_DEV_HARD_NEGATIVE_COSINE',
        'folds_completed':4,'rail_holdouts':8,'controls':24,
        'native_8_class_mean':v8.mean(native_pos),
        'v10_8_class_mean':v8.mean(v10_pos),
        'native_auc':native_auc,'v10_auc':v10_auc,
        'native_gap':native_gap,'v10_gap':v10_gap,
        'native_top1_of_four':native_top1,
        'v10_top1_of_four':v10_top1,
        'native_control_mean':v8.mean(native_ctrl),
        'v10_control_mean':v8.mean(v10_ctrl),
        'target_0_70_met':mean_gate,'control_gate_met':control_gate,
        'outer_control_rule':'v10 AUC >= max(0.70, matched native AUC - 0.03); gap >= max(0.15, native gap - 0.05); top1 >= native top1 - 1',
        'selected_projection_folds':sum(s['selected_config'] is not None for s in summaries),
        'native_fallback_folds':sum(s['selected_config'] is None for s in summaries),
        'inner_dev_pairs_per_fold':[len(f['dev']) for f in folds],
        'configuration_count':len(cfgs),'native_epochs':native_epochs,
        'projection_epochs':proj_epochs,'seed':seed,
        'target_edges_used_for_selection':False,
        'training_data_type':'GCI0 named subclass only',
        'named_subclass_leakage_detected':False,
        'full_owl_semantic_entailment_checked':False,
        'named_only_hermit_audit_previously_passed':True,
        'assumed_semantic_conflicts_absent_not_verified':True,
        'hard_negatives_are_not_owl_proven_false':True,
        'baseline_v9_reference':reference['v9_metric_summary'],
        'matched_native_seed_design_not_guaranteed_bit_identical':True,
        'original_project6_pipeline_modified':False,
        'inputs_sha256':snapshot,'review_file_sha256':v8.digest(review)}
    if v8.immutability_snapshot(root)!=snapshot:
        raise RuntimeError('Project 6 original inputs mutated during v10 run')
    check_reference_unchanged(root,reference)
    v8.save_json(report_dir/'eight_class_summary.json',summary)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--reviewed-anchors',type=Path,default=v9.DEFAULT_REVIEW)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--native-epochs',type=int,default=1000)
    p.add_argument('--projection-epochs',type=int,default=100)
    p.add_argument('--configs',type=int,default=8)
    p.add_argument('--dev-pairs',type=int,default=40)
    p.add_argument('--seed',type=int,default=101)
    a=p.parse_args(argv)
    if a.native_epochs<1 or a.projection_epochs<1 or not 1<=a.configs<=8 or a.dev_pairs<8:
        p.error('epochs >= 1, config count 1..8, dev pairs >= 8 required')
    root=a.project_root.resolve()
    try:
        snapshot=v8.immutability_snapshot(root)
        reference=check_frozen_v9(root,snapshot)
        log('Verified completed v9 reference and SHA-256 provenance; v9 remains unchanged')
        inputs=v8.load_split(root)
        review,anchors=v9.check_review(root,a.reviewed_anchors,inputs)
        folds,_=prepare(root,inputs,anchors,snapshot,review,a.seed,a.dev_pairs,reference)
        log('Prepared 4 folds; 7 reviewed anchors; inner/outer named holdouts protected')
        if a.prepare_only:
            log('PREPARED ONLY: reports/experiments/'+TAG+'/preparation.json')
            return 0
        summary=evaluate(root,folds,inputs,anchors,snapshot,review,reference,a.seed,
                         a.native_epochs,a.projection_epochs,a.configs)
        log('Native held-out cosine mean: '+str(summary['native_8_class_mean']))
        log('v10 held-out cosine mean: '+str(summary['v10_8_class_mean']))
        log('Result: '+summary['status'])
        log('Report: reports/experiments/'+TAG+'/eight_class_summary.json')
        return 0
    except Exception as error:
        print('[Rail mOWL v10] FAILED: '+repr(error),file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
