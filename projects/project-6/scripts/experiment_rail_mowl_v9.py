"""Project 6 v9: nested-development-only projection diagnostics (isolated).

Run from projects/project-6:
    python -u scripts/experiment_rail_mowl_v9.py --prepare-only
    python -u scripts/experiment_rail_mowl_v9.py --native-epochs 3 --projection-epochs 3 --configs 3
    python -u scripts/experiment_rail_mowl_v9.py

Inputs are the original, provenance-checked Project 6 Stage 5A graphs and
exactly seven approved anchors from v8. The eight rail targets NEVER enter
outer-fold mOWL training, residual projection gradients, or hyperparameter
selection. Within each outer fold, inner development subclass assertions are
excluded from both inner mOWL and inner projection training. All tuning uses
inner-development examples; each outer test is evaluated only once.

Training uses ONLY named subclass GCI0 axioms. Control pairs are unasserted
alternatives, NOT OWL-verified false assertions. OWL-semantic leakage has NOT
been excluded by a reasoner. Do not claim ontology improvement solely from
cosine or ROC-AUC on 8 tiny held-out pairs.

Requires adjacent scripts/experiment_rail_mowl_v8.py and
scripts/audit_rail_superclass_support.py, rdflib, numpy, torch, mowl, JVM.
Writes ONLY beneath generated/experiments/rail_mowl_v9_reviewed and
reports/experiments/rail_mowl_v9_reviewed. Does not overwrite v8 outputs,
source ontologies, main run_all.py, or the Step5A intermediate files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import experiment_rail_mowl_v8 as v8
import audit_rail_superclass_support as audit

VERSION = 'project6-v9-nested-dev-geometry-protected-v1'
TAG = 'rail_mowl_v9_reviewed'
DEFAULT_REVIEW = Path('reports/experiments/rail_anchor_review/approved_anchors.json')
DEFAULT_DEV_PAIRS = 40
DEV_MIN_PAIRS = 8
PROJECTION_EPOCHS = 100
NATIVE_EPOCHS = 1000


def log(s):
    print('[Rail mOWL v9] '+str(s), flush=True)


def seeded_order(items, seed, tag):
    return sorted(items,key=lambda x:hashlib.sha256((str(seed)+'\0'+tag+'\0'+repr(x)).encode('utf-8')).hexdigest())


def check_review(root, review, inputs):
    if not review.is_absolute():
        review=root/review
    review=review.resolve()
    if not review.is_file():
        raise FileNotFoundError('Seven approved anchor record not found: '+str(review))
    raw=json.loads(review.read_text(encoding='utf-8'))
    rows=raw.get('approved_anchors',[])
    if raw.get('human_review_status')!='SEVEN_APPROVED_FOR_EXPERIMENT_2026_11_19' or not raw.get('target_holdouts_must_remain_excluded'):
        raise ValueError('Requires the explicitly approved, holdout-excluding v8 JSON record')
    if len(rows)!=7 or {(x.get('child'),x.get('parent')) for x in rows}!=v8.APPROVED_SEVEN:
        raise ValueError('Only exactly the seven user-approved superclass axioms are allowed')
    verified=set(audit.validated_anchors(review,inputs))
    if verified!=v8.APPROVED_SEVEN:
        raise ValueError('Independent approved-anchor validator rejected these relationships')
    return review,verified


def choose_inner_dev(outer_edges, eligible, seed, size, min_pairs=DEV_MIN_PAIRS):
    """Remove source/import pairs only, preferring children with another anchor.

    A removed pair must not remain reachable via named subclass transitivity.
    Outer held-out rail edges have already been removed, so this function
    never sees or selects one as a development target.
    """
    selected=[]
    remaining=set(outer_edges)
    candidates=[p for p in seeded_order(eligible,seed,'dev') if p in remaining]
    # Preserve at least one outgoing named parent as the first preference.
    for preferred in (True,False):
        for pair in candidates:
            if len(selected)>=size: break
            if pair not in remaining: continue
            child,_=pair
            if child in {c for c,_ in v8.FOLDS[0]+v8.FOLDS[1]+v8.FOLDS[2]+v8.FOLDS[3]}:
                continue
            if preferred and not any(c==child for c,p in remaining if (c,p)!=pair):
                continue
            proposed=remaining-{pair}
            if pair in v8.closure(proposed):
                continue
            remaining=proposed
            selected.append(pair)
        if len(selected)>=size: break
    if len(selected)<min_pairs:
        raise ValueError(f'Only {len(selected)} usable inner-development edges; minimum is {min_pairs}. No rail holdout used for tuning.')
    if set(selected)&(remaining|v8.closure(remaining)):
        raise RuntimeError('Inner development target direct/transitive leakage detected')
    return remaining,sorted(selected)


def deterministic_controls(full_edges, positives, seed, count=3):
    """Named-unentailed alternative parents; never assert OWL-negatives."""
    ancestor=v8.closure(full_edges)
    parents=sorted({p for _,p in full_edges})
    result={}
    for child, true_parent in positives:
        options=[p for p in parents if p not in (child,true_parent) and (child,p) not in ancestor]
        if len(options)<count:
            raise ValueError(f'Insufficient named-unentailed control candidates for {child}')
        result[(child,true_parent)] = seeded_order(options,seed,child+'|'+true_parent)[:count]
    return result


def build_preparation(root,inputs,anchors,snapshot,review,seed,dev_size):
    all_edges=set(inputs['edges'])|anchors
    classes=set(inputs['classes'])|{x for edge in anchors for x in edge}
    rail_children={c for fold in v8.FOLDS for c,_ in fold}
    # ONLY source/import subclass relationships: no rail child for inner tuning.
    dev_candidates={edge for edge in inputs['edges'] if edge[0] not in rail_children}
    graph_dir=root/'generated/experiments'/TAG
    report_dir=root/'reports/experiments'/TAG
    folds=[]
    for i, holdout in enumerate(v8.FOLDS,1):
        outer=set(all_edges)-set(holdout)
        if set(holdout)&(outer|v8.closure(outer)):
            raise ValueError(f'Outer fold {i}: direct/transitive named subclass leakage')
        inner,dev=choose_inner_dev(outer,dev_candidates,seed+1000*i,dev_size)
        if any(x in inner or x in v8.closure(inner) for x in dev):
            raise RuntimeError(f'Fold {i}: inner dev named entailment leakage')
        if not anchors.issubset(inner):
            raise RuntimeError('Inner tuning removed an approved rail superclass anchor')
        if set(holdout)&(inner|v8.closure(inner)):
            raise RuntimeError('Outer holdout leaked into the inner training graph')
        for kind, edges in (('outer',outer),('inner',inner)):
            p=graph_dir/f'fold_{i}_{kind}_train.ttl'
            v8.save_training_graph(p,edges,classes)
            if v8.named_pairs(v8.graph_load(p))!=edges:
                raise RuntimeError('Failed to round-trip saved training graph')
        outer_path=graph_dir/f'fold_{i}_outer_train.ttl'
        inner_path=graph_dir/f'fold_{i}_inner_train.ttl'
        support={c:v8.determine_support(outer,c) for c,_ in holdout}
        if any(s['training_outgoing']==0 for s in support.values()):
            raise RuntimeError('Unexpected missing reviewed superclass anchor')
        folds.append({'fold':i,'outer_edges':outer,'inner_edges':inner,
                      'outer_path':outer_path,'inner_path':inner_path,
                      'outer_sha256':v8.digest(outer_path),'inner_sha256':v8.digest(inner_path),
                      'heldout':list(holdout),'dev':dev,'support':support})
    manifest={'version':VERSION,'mode':'SEVEN_APPROVED_ANCHORS_NESTED_DEVELOPMENT',
              'status':'PREPARED_NO_TRAINING','approved_anchor_count':7,
              'review_file_sha256':v8.digest(review),'inputs_sha256':snapshot,
              'outer_holdout_total':8,'outer_folds':4,'dev_max_requested':dev_size,
              'named_subclass_leakage_detected':False,
              'full_owl_semantic_entailment_checked':False,
              'outer_tests_not_used_for_selection':True,
              'development_selection_source':'source/import edges only; never rail child',
              'folds':[{'fold':f['fold'],'outer_training_pairs':len(f['outer_edges']),
                        'inner_training_pairs':len(f['inner_edges']),'dev_pairs':len(f['dev']),
                        'outer_sha256':f['outer_sha256'],'inner_sha256':f['inner_sha256'],
                        'outer_training_path':f['outer_path'].relative_to(root).as_posix(),
                        'inner_training_path':f['inner_path'].relative_to(root).as_posix(),
                        'heldout':[{'child':c,'parent':p,**f['support'][c]} for c,p in f['heldout']],
                        'dev_hash':hashlib.sha256(json.dumps(f['dev']).encode()).hexdigest()}
                       for f in folds]}
    if v8.immutability_snapshot(root)!=snapshot:
        raise RuntimeError('Upstream Project 6 files changed during v9 preparation')
    v8.save_json(report_dir/'preparation.json',manifest)
    return folds,manifest


def configs(epochs):
    # Fixed BEFORE outer evaluations. The v8 reference is a candidate but
    # must win on internal development, not outer rail test performance.
    presets=[
        ('v8_reference', .25,.10,0.0,0.00),
        ('identity_0p5',.25,.50,0.0,.05),
        ('identity_1',.25,1.00,0.0,.10),
        ('identity_2',.10,2.00,0.0,.10),
        ('identity_4',.10,4.00,0.0,.20),
        ('gentle_projection',.05,2.00,0.0,.20),
        ('contrastive_light',.10,1.00,.10,.10),
        ('contrastive_medium',.10,2.00,.25,.15),
        ('contrastive_conservative',.05,4.00,.25,.20),
        ('contrastive_strong',.10,2.00,.50,.20),
    ]
    return [dict(name=name,epochs=epochs,hidden_size=32,learning_rate=.0025,
                 residual_scale=scale,identity_weight=id_weight,
                 contrastive_weight=contrastive,geometry_weight=geometry,
                 margin=.10,temperature=.25,weight_decay=1e-5,
                 positive_weight=1.0,residual_penalty_weight=.001)
            for name,scale,id_weight,contrastive,geometry in presets]


def sample_negative_pairs(training,index,seed,exclude_children=(),per_positive=1):
    """Unasserted parent alternatives, screened by named closure only.

    Avoid ALL rail children and inner dev children to prevent validation
    control pairs ever entering negative gradients; no claimed OWL falsity.
    """
    excluded=set(exclude_children)
    excluded|={c for fold in v8.FOLDS for c,_ in fold}
    candidates=sorted({p for _,p in training if p in index})
    reachable=v8.closure(training)
    negatives=[]
    for j,(c,p) in enumerate(sorted(training)):
        if c in excluded or c not in index or p not in index:continue
        choices=[x for x in candidates if x!=c and x!=p and (c,x) not in reachable]
        for alt in seeded_order(choices,seed,'trainneg:'+str(j))[:per_positive]:
            negatives.append((c,p,alt))
    return negatives


def geometry_pairs(index,seed,count=512):
    ids=sorted(index)
    if len(ids)<3: return []
    selected=[]
    for n in range(count):
        j=seeded_order(ids,seed+n,'geometry')
        if j[0]!=j[1]:selected.append((index[j[0]],index[j[1]]))
    return selected


def geometry_report(native,projected,draw):
    import torch
    import torch.nn.functional as F
    if not draw:return {'native_offdiag_mean':None,'projected_offdiag_mean':None,'mean_absolute_shift':None}
    aa=torch.tensor([i for i,j in draw],dtype=torch.long)
    bb=torch.tensor([j for i,j in draw],dtype=torch.long)
    with torch.no_grad():
        x=F.normalize(native,dim=1)
        y=F.normalize(projected,dim=1)
        orig=(x[aa]*x[bb]).sum(dim=1)
        later=(y[aa]*y[bb]).sum(dim=1)
        return {'native_offdiag_mean':float(orig.mean()),'projected_offdiag_mean':float(later.mean()),
                'mean_absolute_shift':float((later-orig).abs().mean()),
                'mean_signed_shift':float((later-orig).mean()),
                'native_offdiag_std':float(orig.std(unbiased=False)),
                'projected_offdiag_std':float(later.std(unbiased=False))}


def fit_projection(base,index,training,config,seed,negative_rows,geometry_draw):
    import torch
    import torch.nn.functional as F
    v8.seed_everything(seed)
    model=v8.make_projection(base.shape[1],config['hidden_size'],config['residual_scale']).to('cpu')
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    positives=torch.tensor([(index[c],index[p]) for c,p in sorted(training)],dtype=torch.long)
    used=torch.unique(positives.reshape(-1))
    x=F.normalize(base,dim=1)
    negatives=torch.tensor([(index[c],index[p],index[n]) for c,p,n in negative_rows],dtype=torch.long) if negative_rows else None
    geo_left=torch.tensor([a for a,b in geometry_draw],dtype=torch.long) if geometry_draw else None
    geo_right=torch.tensor([b for a,b in geometry_draw],dtype=torch.long) if geometry_draw else None
    with torch.no_grad():
        geo_native=(x[geo_left]*x[geo_right]).sum(dim=1) if geometry_draw else None
    best_state=None
    best_train_loss=float('inf')
    final_loss=None
    for epoch in range(1,config['epochs']+1):
        opt.zero_grad()
        y=model(base)
        pos_sim=(y[positives[:,0]]*y[positives[:,1]]).sum(dim=1)
        pos_loss=(1-pos_sim).mean()
        id_loss=(1-(y[used]*x[used]).sum(dim=1)).mean()
        residual=model.fc2(model.activation(model.fc1(base[used])))
        resid_loss=(residual**2).mean()
        loss=(config['positive_weight']*pos_loss
              +config['identity_weight']*id_loss
              +config['residual_penalty_weight']*resid_loss)
        if config['geometry_weight'] and geometry_draw:
            geo_post=(y[geo_left]*y[geo_right]).sum(dim=1)
            loss=loss+config['geometry_weight']*(geo_post-geo_native).square().mean()
        if config['contrastive_weight'] and negatives is not None and len(negatives):
            pos=(y[negatives[:,0]]*y[negatives[:,1]]).sum(dim=1)
            neg=(y[negatives[:,0]]*y[negatives[:,2]]).sum(dim=1)
            contrastive=F.softplus((neg-pos+config['margin'])/config['temperature']).mean()
            loss=loss+config['contrastive_weight']*contrastive
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
        opt.step()
        value=float(loss.detach())
        if not math.isfinite(value):raise RuntimeError('Nonfinite projection loss')
        if value<best_train_loss:
            best_train_loss=value
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        final_loss=value
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): y=model(base)
    return model,y,{'best_training_loss':best_train_loss,'last_training_loss':final_loss}


def stats(mat,index,truth,control_map):
    true_scores=v8.score(mat,truth,index)
    false_scores=[]
    pairwise_wins=[]
    rows=[]
    for (child,parent),score in zip(truth,true_scores):
        alt=[(child,p) for p in control_map[(child,parent)]]
        scores=v8.score(mat,alt,index)
        false_scores.extend(scores)
        pairwise_wins.extend([1.0 if score>x else 0.5 if score==x else 0.0 for x in scores])
        rows.append({'child':child,'expected_parent':parent,'cosine':score,
                     'control_mean':float(np.mean(scores)),
                     'control_margin':score-float(np.mean(scores)),
                     'rank_among_four':1+sum(x>score for x in scores),
                     'controls':list(zip(control_map[(child,parent)],scores))})
    return {'positive_mean':v8.mean(true_scores),'control_mean':v8.mean(false_scores),
            'gap':v8.mean(true_scores)-v8.mean(false_scores),
            'auc':v8.rank_auc(true_scores,false_scores),
            'paired_win_rate':float(np.mean(pairwise_wins)),
            'top1_of_four':sum(x['rank_among_four']==1 for x in rows),'count':len(truth),
            'rows':rows}


def acceptable(candidate, baseline):
    """A-priori dev-only guards; do not relax after seeing outer test results."""
    return (candidate['auc']>=max(.70,baseline['auc']-.015)
            and candidate['gap']>=max(.15,baseline['gap']-.025)
            and candidate['geometry']['mean_absolute_shift']<=.12)


def select_config(eligible, baseline):
    # No outer rail scores appear here. Require a material *dev* improvement;
    # otherwise keep frozen native geometry intact.
    choices=[r for r in eligible if r['passes_dev_guards'] and
             (r['dev_auc']>baseline['auc']+.01 or r['dev_gap']>baseline['gap']+.02)]
    if not choices:return None
    return max(choices,key=lambda r:(round(r['dev_auc'],6),round(r['dev_gap'],6),
                                    -r['geometry_mean_absolute_shift'],r['config']['name']))


def save_rows(root, rows):
    # Flat CSV helps independent human review and avoids nested control JSON.
    cols=['fold','record_type','child','expected_parent','control_parent',
          'native_cosine','v9_cosine','native_control_mean','v9_control_mean',
          'native_rank_among_four','v9_rank_among_four','selection']
    v8.save_csv(root/'all_pair_scores.csv',rows,cols)


def evaluate(root,folds,inputs,anchors,snapshot,review,seed,native_epochs,proj_epochs,cfg_count):
    import torch
    import mowl
    mowl.init_jvm('2g')
    report_dir=root/'reports/experiments'/TAG
    graph_dir=root/'generated/experiments'/TAG
    cfgs=configs(proj_epochs)[:cfg_count]
    all_rows=[]
    folded=[]
    overall_native=[];overall_v9=[]
    all_development=[]
    for f in folds:
        i=f['fold']
        for kind in ('outer','inner'):
            if v8.digest(f[kind+'_path'])!=f[kind+'_sha256']:
                raise RuntimeError('Training fold file hash changed after preparation')
        log(f'Fold {i}/4: training INNER development ELEmbeddings ({native_epochs} epochs)')
        idx_dev,base_dev,_=v8.fit_native(f['inner_path'],seed+i*11,native_epochs,v8.EMBED_DIM)
        dev_control=deterministic_controls(f['outer_edges'],f['dev'],seed+i*41)
        required_dev={a for pair in f['dev'] for a in pair}
        required_dev|={c for v in dev_control.values() for c in v}
        if required_dev-set(idx_dev):
            raise RuntimeError('Inner development index misses validation classes')
        base_dev_stats=stats(base_dev,idx_dev,f['dev'],dev_control)
        dev_neg=sample_negative_pairs(f['inner_edges'],idx_dev,seed+i*33,
                                      exclude_children={c for c,p in f['dev']})
        draw_dev=geometry_pairs(idx_dev,seed+i*202)
        candidates=[]
        log(f'Fold {i}/4: tuning {len(cfgs)} projection configurations on INNER source/import development edges')
        for k,cfg in enumerate(cfgs):
            _,projected,loss=fit_projection(base_dev,idx_dev,f['inner_edges'],cfg,seed+i*100+k,
                                            dev_neg,draw_dev)
            dev=stats(projected,idx_dev,f['dev'],dev_control)
            geo=geometry_report(base_dev,projected,draw_dev)
            passes=acceptable({**dev,'geometry':geo},base_dev_stats)
            row={'fold':i,'config':cfg,'dev_auc':dev['auc'],'dev_gap':dev['gap'],
                 'dev_paired_win_rate':dev['paired_win_rate'],
                 'dev_positive_mean':dev['positive_mean'],'dev_control_mean':dev['control_mean'],
                 'geometry_mean_absolute_shift':geo['mean_absolute_shift'],
                 'passes_dev_guards':passes,'loss':loss}
            candidates.append(row)
            all_development.append(row)
        selected=select_config(candidates,base_dev_stats)
        chosen=selected['config'] if selected else None
        fold_dev={'fold':i,'development_target_count':len(f['dev']),
                  'native_dev':{k:v for k,v in base_dev_stats.items() if k!='rows'},
                  'configs':candidates,'selected_config':chosen,
                  'selection_rule':'predeclared development AUC, gap, geometric distortion; never outer rail tests',
                  'fallback_native':not bool(selected),
                  'all_dev_targets_excluded_from_inner_native_and_projection_training':True}
        v8.save_json(report_dir/f'fold_{i}_development.json',fold_dev)
        log(f'Fold {i}/4: development selection = '+(chosen['name'] if chosen else 'NATIVE_ONLY'))
        log(f'Fold {i}/4: training OUTER ELEmbeddings ({native_epochs} epochs)')
        idx,base,versions=v8.fit_native(f['outer_path'],seed+i*11+1,native_epochs,v8.EMBED_DIM)
        if any(c not in idx or p not in idx for c,p in f['heldout']):
            raise RuntimeError('Outer holdout classes not in native mOWL index')
        control_iris=v8.controls(set(inputs['edges'])|anchors,tuple(f['heldout']))
        cmap={(c,p):control_iris[c] for c,p in f['heldout']}
        if any(q not in idx for v in cmap.values() for q in v):
            raise RuntimeError('Outer control classes not in native mOWL index')
        base_outer=stats(base,idx,f['heldout'],cmap)
        # Only now: inner development has selected the hyperparameters.
        draw_outer=geometry_pairs(idx,seed+i*808)
        fit_details=None
        if chosen:
            neg=sample_negative_pairs(f['outer_edges'],idx,seed+i*52)
            model,after,fit_details=fit_projection(base,idx,f['outer_edges'],chosen,
                                                  seed+i*100+1000,neg,draw_outer)
            # Save only the chosen projector, with provenance and selection data.
            torch.save({'state_dict':model.state_dict(),'config':chosen,'fold':i,
                        'train_sha256':f['outer_sha256'],
                        'chosen_on':'inner_source_import_development_only'},
                       graph_dir/f'fold_{i}_selected_projection.pt')
        else:
            after=base.clone()
        improved_outer=stats(after,idx,f['heldout'],cmap)
        geo_outer=geometry_report(base,after,draw_outer)
        overall_native.extend(base_outer['rows'])
        overall_v9.extend(improved_outer['rows'])
        for n,v in zip(base_outer['rows'],improved_outer['rows']):
            all_rows.append({'fold':i,'record_type':'HELDOUT','child':n['child'],
                             'expected_parent':n['expected_parent'],
                             'native_cosine':n['cosine'],'v9_cosine':v['cosine'],
                             'native_control_mean':n['control_mean'],
                             'v9_control_mean':v['control_mean'],
                             'native_rank_among_four':n['rank_among_four'],
                             'v9_rank_among_four':v['rank_among_four'],
                             'selection':chosen['name'] if chosen else 'NATIVE_ONLY'})
            for (cp,nscore),(cp2,vscore) in zip(n['controls'],v['controls']):
                assert cp==cp2
                all_rows.append({'fold':i,'record_type':'UNASSERTED_PARENT_CONTROL',
                                 'child':n['child'],'control_parent':cp,
                                 'native_cosine':nscore,'v9_cosine':vscore,
                                 'selection':chosen['name'] if chosen else 'NATIVE_ONLY'})
        fold_summary={'fold':i,'selection':chosen['name'] if chosen else 'NATIVE_ONLY',
                      'selected_config':chosen,'outer_native':base_outer,
                      'outer_v9':improved_outer,'outer_geometry':geo_outer,
                      'native_library_versions':versions,'final_fit_details':fit_details,
                      'heldout_not_used_to_select_config':True,
                      'outer_training_hash':f['outer_sha256'],
                      'inner_training_hash':f['inner_sha256']}
        v8.save_json(report_dir/f'fold_{i}_metrics.json',fold_summary)
        ordered=sorted(idx)
        with (graph_dir/f'fold_{i}_vectors.npz').open('wb') as handle:
            np.savez_compressed(handle,iris=np.asarray(ordered,dtype='U'),
                native=np.asarray([base[idx[s]].numpy() for s in ordered],dtype=np.float32),
                v9=np.asarray([after[idx[s]].numpy() for s in ordered],dtype=np.float32))
        folded.append(fold_summary)
        log(f"Fold {i}: native {base_outer['positive_mean']:.4f}; v9 {improved_outer['positive_mean']:.4f}; "
            f"v9 gap {improved_outer['gap']:.4f}; native AUC {base_outer['auc']:.3f}; v9 AUC {improved_outer['auc']:.3f}")
    save_rows(report_dir,all_rows)
    positive=[r for r in all_rows if r['record_type']=='HELDOUT']
    alternative=[r for r in all_rows if r['record_type']=='UNASSERTED_PARENT_CONTROL']
    vals=lambda key,rr:[float(r[key]) for r in rr]
    npos=vals('native_cosine',positive);npos_ctrl=vals('native_cosine',alternative)
    vpos=vals('v9_cosine',positive);vpos_ctrl=vals('v9_cosine',alternative)
    gap=v8.mean(vpos)-v8.mean(vpos_ctrl)
    auc=v8.rank_auc(vpos,vpos_ctrl)
    mean_gate=v8.mean(vpos)>=v8.TARGET
    controls_gate=(gap>=.15 and auc>=.7)
    summary={'version':VERSION,'status':('TARGET_AND_CONTROLS_MET_DIAGNOSTIC_ONLY' if mean_gate and controls_gate
                                         else 'TARGET_OR_CONTROLS_NOT_MET'),
             'experiment_mode':'NESTED_DEV_SEVEN_APPROVED_ANCHORS',
             'outer_test_sample_size':8,'folds_completed':4,
             'native_8_class_mean':v8.mean(npos),'v9_8_class_mean':v8.mean(vpos),
             'native_control_mean':v8.mean(npos_ctrl),'v9_control_mean':v8.mean(vpos_ctrl),
             'native_auc':v8.rank_auc(npos,npos_ctrl),'v9_auc':auc,
             'native_gap':v8.mean(npos)-v8.mean(npos_ctrl),'v9_gap':gap,
             'native_top1_of_four':sum(r['native_rank_among_four']==1 for r in positive),
             'v9_top1_of_four':sum(r['v9_rank_among_four']==1 for r in positive),
             'projection_selected_on_development_folds':sum(f['selected_config'] is not None for f in folded),
             'projection_fallback_to_native_folds':sum(f['selected_config'] is None for f in folded),
             'configs_per_fold':len(cfgs),'dev_pairs_per_fold':[len(f['dev']) for f in folds],
             'mean_threshold':v8.TARGET,'target_0_70_met':mean_gate,'control_gate_met':controls_gate,
             'all_8_rail_targets_unseen_during_model_selection':True,
             'all_inner_development_targets_unseen_by_inner_mowl_and_projection':True,
             'negative_sampling':'optional, only for contrastive candidates; named-unentailed alternative parents, never OWL proven false',
             'only_gci0_named_subclass_training':True,
             'full_owl_semantic_entailment_checked':False,
             'outer_named_subclass_leakage_detected':False,
             'development_named_subclass_leakage_detected':False,
             'comparison_to_v8':'do not treat a development-selected v9 and v8 as matched runs without accounting for different seeds and sampling',
             'baseline_reference_v8_native_mean':.5624942779541016,
             'baseline_reference_v8_projected_mean':.9385036453604698,
             'baseline_reference_v8_projected_auc':.8802083333333334,
             'input_sha256':snapshot,'review_file_sha256':v8.digest(review),
             'original_pipeline_modified':False,'further_validation_required':
                 'Full OWL reasoner entailment check, more known-positive holdouts, independent negatives/controls, uncertainty estimates'}
    if v8.immutability_snapshot(root)!=snapshot:
        raise RuntimeError('Project 6 source/pipeline inputs mutated during v9 training')
    v8.save_json(report_dir/'eight_class_summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--reviewed-anchors',type=Path,default=DEFAULT_REVIEW)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--native-epochs',type=int,default=NATIVE_EPOCHS)
    p.add_argument('--projection-epochs',type=int,default=PROJECTION_EPOCHS)
    p.add_argument('--configs',type=int,default=10)
    p.add_argument('--dev-pairs',type=int,default=DEFAULT_DEV_PAIRS)
    p.add_argument('--seed',type=int,default=101)
    a=p.parse_args()
    if a.native_epochs<1 or a.projection_epochs<1 or not 1<=a.configs<=10 or a.dev_pairs<DEV_MIN_PAIRS:
        p.error('epochs must be >=1, 1<=configs<=10, and dev-pairs>=8')
    root=a.project_root.resolve()
    try:
        snapshot=v8.immutability_snapshot(root)
        log('Loading immutable v8 Step 5A sources and eight audited rail holdouts')
        inputs=v8.load_split(root)
        review,anchors=check_review(root,a.reviewed_anchors,inputs)
        folds,manifest=build_preparation(root,inputs,anchors,snapshot,review,a.seed,a.dev_pairs)
        log('Prepared 4 isolated outer/inner folds; 7 approved anchors; named leakage checks PASS')
        if a.prepare_only:
            log('PREPARED ONLY: reports/experiments/'+TAG+'/preparation.json')
            return 0
        log('Hyperparameters selected only on inner development pairs; outer rail targets are unseen')
        summary=evaluate(root,folds,inputs,anchors,snapshot,review,a.seed,a.native_epochs,
                         a.projection_epochs,a.configs)
        log('Native mean: '+str(summary['native_8_class_mean']))
        log('v9 mean: '+str(summary['v9_8_class_mean']))
        log('Result: '+summary['status'])
        log('Reports: reports/experiments/'+TAG+'/eight_class_summary.json')
        return 0
    except Exception as exc:
        print('RAIL MOWL V9 FAILED: '+repr(exc),file=sys.stderr,flush=True)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
