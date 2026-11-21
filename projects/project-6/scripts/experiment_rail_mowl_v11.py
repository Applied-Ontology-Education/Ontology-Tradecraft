"""Final one-command Project 6 cosine experiment: native ELEmbeddings + one-hop anchors.

Run ONCE from projects/project-6 using your existing .venv-mowl:
    python -u scripts/experiment_rail_mowl_v11.py

No separate prepare or smoke test. The script prepares isolated training graphs,
selects ONE global hyperparameter/one-hop parent-aggregation setting on 160
source/import development assertions (40 in each outer fold), and evaluates the
original eight rail holdouts once, across the four frozen folds.

It compares against matched-seed native v9-style ELEmbeddings (50 dimensions,
1000 epochs) and records the previously frozen v9 metrics. The candidate
one-hop aggregation is based exclusively on observed training subClassOf edges.
It does not import properties or fabricate OWL EL restrictions, does not apply
arbitrary cosine recalibration, does not use rail labels to tune the model,
and does not modify source, v8, v9, v10, or run_all.py.

This is an *exploratory* final attempt on the repeatedly inspected eight rail
holdouts, not an independent generalization test. Named-only HermiT passed;
complete imported-OWL closure was not verified.

Requires existing scripts experiment_rail_mowl_v8.py, v9.py, v10.py;
installed mowl, torch, numpy, rdflib, and local Project 6 inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import experiment_rail_mowl_v8 as v8
import experiment_rail_mowl_v9 as v9
import experiment_rail_mowl_v10 as v10

TAG = 'rail_mowl_v11_native_graph_cosine'
VERSION = 'project6-v11-final-single-run-native-graph-cosine-v1'
SEED = 101
DEV_PER_FOLD = 40
DIMENSIONS = (50, 40, 75, 100)
MIXES = (0.0, 0.15, 0.30)

# Frozen, predeclared. Each configuration uses the same seeds/splits across
# folds. The reference is precisely the v9 native training specification.
NATIVE_CONFIGS = (
    {'name': 'v9_native_reference', 'dim': 50, 'epochs': 1000, 'margin': 0.10, 'learning_rate': 0.0010, 'reg_norm': 1.0},
    {'name': 'compact_40',         'dim': 40, 'epochs': 1200, 'margin': 0.10, 'learning_rate': 0.0010, 'reg_norm': 1.0},
    {'name': 'wide_75',            'dim': 75, 'epochs': 1200, 'margin': 0.10, 'learning_rate': 0.0010, 'reg_norm': 1.0},
    {'name': 'wide_100',           'dim':100, 'epochs': 1200, 'margin': 0.10, 'learning_rate': 0.0010, 'reg_norm': 1.0},
    {'name': 'small_margin_50',    'dim': 50, 'epochs': 1200, 'margin': 0.05, 'learning_rate': 0.0010, 'reg_norm': 1.0},
    {'name': 'small_margin_75',    'dim': 75, 'epochs': 1200, 'margin': 0.05, 'learning_rate': 0.0010, 'reg_norm': 1.0},
)


def log(s):
    print('[Rail mOWL v11] ' + str(s), flush=True)


def fit_native_config(path, seed, config):
    """ELEmbeddings without touching existing native fit implementation."""
    import torch
    import mowl
    from mowl.datasets import Dataset, PathDataset
    from mowl.models import ELEmbeddings

    v8.seed_everything(seed)
    dataset = Dataset(PathDataset(str(path.resolve())).ontology)
    model = ELEmbeddings(dataset,
                         embed_dim=config['dim'],
                         margin=config['margin'],
                         reg_norm=config['reg_norm'],
                         learning_rate=config['learning_rate'],
                         batch_size=128,
                         model_filepath=None, device='cpu')
    idx = {str(iri): int(n) for iri, n in model.class_index_dict.items()}
    model.train(epochs=config['epochs'], validate_every=config['epochs'])
    matrix = model.module.class_embed.weight.detach().cpu().clone()
    if matrix.shape != (len(idx), config['dim']) or not torch.isfinite(matrix).all():
        raise RuntimeError('Invalid native embedding dimensions or non-finite values')
    return idx, matrix, {'mowl_version': getattr(mowl, '__version__', 'unknown'),
                         'torch_version': torch.__version__}


def aggregate_parents(base, index, training_edges, alpha):
    """One-hop only; use *observed TRAINING* parents; never the heldout parent."""
    import torch
    import torch.nn.functional as F

    if not 0.0 <= alpha <= 0.30:
        raise ValueError('Parent aggregation weight outside predeclared bound')
    unit = F.normalize(base.detach(), p=2, dim=1)
    if alpha == 0:
        return unit.clone()
    parents = defaultdict(set)
    for child, parent in training_edges:
        if child != parent and child in index and parent in index:
            parents[child].add(parent)
    output = unit.clone()
    with torch.no_grad():
        for child, ps in sorted(parents.items()):
            indices = [index[x] for x in sorted(ps)]
            avg = F.normalize(unit[indices].mean(dim=0), dim=0)
            output[index[child]] = F.normalize((1-alpha)*unit[index[child]] + alpha*avg, dim=0)
    return output


def pool_stats(stats_list):
    """Pool positive/control score rows across all four folds."""
    positive, alternative = [], []
    ranks, wins = [], []
    for result in stats_list:
        for row in result['rows']:
            p = float(row['cosine'])
            positive.append(p)
            ranks.append(int(row['rank_among_four']))
            for _, neg in row['controls']:
                q = float(neg)
                alternative.append(q)
                wins.append(1.0 if p > q else 0.5 if p == q else 0.0)
    return {'positive_mean': v8.mean(positive), 'control_mean': v8.mean(alternative),
            'gap': v8.mean(positive)-v8.mean(alternative),
            'auc': v8.rank_auc(positive, alternative),
            'paired_win_rate': float(np.mean(wins)),
            'top1_of_four': sum(x == 1 for x in ranks),
            'positive_count': len(positive), 'control_count': len(alternative)}


def dev_guard(candidate, baseline):
    """Predeclared minimum cosine improvement with discrimination safeguards."""
    return (candidate['positive_mean'] >= baseline['positive_mean'] + 0.01
            and candidate['auc'] >= baseline['auc'] - 0.01
            and candidate['gap'] >= baseline['gap'] - 0.015
            and candidate['paired_win_rate'] >= baseline['paired_win_rate'] - 0.015
            and candidate['top1_of_four'] >= baseline['top1_of_four'] - 2
            and candidate['geometry_shift'] <= 0.09)


def choose_global(candidates, baseline):
    """One global setting, chosen BEFORE scoring any rail test pair."""
    qualified = [row for row in candidates if dev_guard(row, baseline)]
    if not qualified:
        return {'name': 'v9_native_reference', 'mix': 0.0}, False
    selected = max(qualified, key=lambda row: (
        row['positive_mean']-baseline['positive_mean'],
        row['auc']-baseline['auc'],
        row['gap']-baseline['gap'],
        -row['geometry_shift'], row['name']))
    return {'name': selected['name'], 'mix': selected['mix']}, True


def build_folds(root, inputs, anchors, snapshot, review):
    before = v9.TAG
    try:
        v9.TAG = TAG
        folds, manifest = v9.build_preparation(root, inputs, anchors, snapshot, review,
                                                SEED, DEV_PER_FOLD)
    finally:
        v9.TAG = before
    manifest['version'] = VERSION
    manifest['mode'] = 'NATIVE_HYPERPARAMETERS_AND_ONE_HOP_TAXONOMY'
    manifest['full_owl_semantic_entailment_checked'] = False
    v8.save_json(root/'reports/experiments'/TAG/'preparation.json', manifest)
    return folds


def aggregate_development(evaluations):
    # evaluations[candidate_key] = list of {'stats', 'shift', 'fold'}
    results = []
    for (name, mix), records in evaluations.items():
        pooled = pool_stats([r['stats'] for r in records])
        if len(records) != 4 or pooled['positive_count'] != 160:
            raise RuntimeError('Incomplete inner-development evaluation')
        results.append({'name': name, 'mix': mix, **pooled,
                        'geometry_shift': float(np.mean([r['shift'] for r in records])),
                        'folds': [{'fold': r['fold'],
                                   'positive_mean': r['stats']['positive_mean'],
                                   'auc': r['stats']['auc'], 'gap': r['stats']['gap']}
                                  for r in records]})
    return sorted(results, key=lambda x: (x['name'], x['mix']))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--reviewed-anchors', type=Path, default=v9.DEFAULT_REVIEW)
    args = ap.parse_args(argv)
    root = args.project_root.resolve()
    report = root/'reports/experiments'/TAG
    graph_dir = root/'generated/experiments'/TAG
    try:
        import mowl
        snapshot = v8.immutability_snapshot(root)
        reference = v10.check_frozen_v9(root, snapshot)
        log('Frozen v9 checksums and eight-class baseline verified')
        inputs = v8.load_split(root)
        review, anchors = v9.check_review(root, args.reviewed_anchors, inputs)
        folds = build_folds(root, inputs, anchors, snapshot, review)
        log('Four isolated outer/inner folds prepared; no named subclass leakage')
        mowl.init_jvm('2g')

        # Development-only model selection. 6 fits * 4 inner folds.
        evaluations = defaultdict(list)
        for f in folds:
            i = f['fold']
            for kind in ('inner', 'outer'):
                if v8.digest(f[kind+'_path']) != f[kind+'_sha256']:
                    raise RuntimeError('Prepared fold graph changed')
            controls = v9.deterministic_controls(f['outer_edges'], f['dev'], SEED+i*41)
            expected = {x for child, parent in f['dev'] for x in (child, parent)}
            expected.update(p for opts in controls.values() for p in opts)
            per_fold = []
            for cfg in NATIVE_CONFIGS:
                log(f"Fold {i}/4 INNER: {cfg['name']} (dim={cfg['dim']}, {cfg['epochs']} epochs)")
                index, matrix, _ = fit_native_config(f['inner_path'], SEED+i*11, cfg)
                if expected - set(index):
                    raise RuntimeError('Model class index omits inner development classes')
                geometry_draw = v9.geometry_pairs(index, SEED+i*202)
                for mix in MIXES:
                    post = aggregate_parents(matrix, index, f['inner_edges'], mix)
                    scores = v9.stats(post, index, f['dev'], controls)
                    if mix == 0:
                        shift = 0.0
                    else:
                        shift = v9.geometry_report(matrix, post, geometry_draw)['mean_absolute_shift']
                    evaluations[(cfg['name'], mix)].append({'fold': i,
                                                               'stats': scores,
                                                               'shift': shift})
                    per_fold.append({'name': cfg['name'], 'mix': mix,
                                     'positive_mean': scores['positive_mean'],
                                     'auc': scores['auc'], 'gap': scores['gap'],
                                     'geometry_shift': shift})
            v8.save_json(report/f'fold_{i}_development.json', {
                'fold': i, 'source_import_dev_pairs': len(f['dev']),
                'outer_rail_test_pairs_used_for_tuning': 0,
                'candidate_results': per_fold})

        dev_candidates = aggregate_development(evaluations)
        base_dev = next(row for row in dev_candidates if row['name']=='v9_native_reference'
                        and row['mix']==0.0)
        choice, upgraded = choose_global(dev_candidates, base_dev)
        for row in dev_candidates:
            row['passes_predeclared_development_guard'] = dev_guard(row, base_dev)
        v8.save_json(report/'development_selection.json', {
            'selection_source': '160 SOURCE/IMPORT relationships; no eight rail test relationships',
            'development_baseline': base_dev, 'candidates': dev_candidates,
            'selected_setting': choice, 'selected_nonbaseline': upgraded,
            'selection_policy': 'maximize dev cosine; AUC loss <= .01, gap loss <= .015, '
                                'paired win loss <= .015, top1 loss <= 2/160, '
                                'geometry shift <= .09; positive cosine gain >= .01'})
        log('GLOBAL development selection: '+choice['name']+' + mix '+str(choice['mix']))

        selected_config = next(c for c in NATIVE_CONFIGS if c['name']==choice['name'])
        reference_config = NATIVE_CONFIGS[0]
        all_rows, fold_metrics, native_results, chosen_results = [], [], [], []

        # Exactly once per outer rail fold, after all hyperparameters are frozen.
        for f in folds:
            i = f['fold']
            log(f'Fold {i}/4 OUTER: v9 native reference (50d, 1000 epochs)')
            idx, base, versions = fit_native_config(f['outer_path'], SEED+i*11+1,
                                                     reference_config)
            holdout = f['heldout']
            sampled = v8.controls(set(inputs['edges'])|anchors, tuple(holdout))
            cmap = {(c,p): sampled[c] for c,p in holdout}
            need = {x for c,p in holdout for x in (c,p)}
            need.update(q for opts in cmap.values() for q in opts)
            if need-set(idx):
                raise RuntimeError('Native reference model missing heldout/control classes')
            native_stats = v9.stats(base, idx, holdout, cmap)
            if choice['name']=='v9_native_reference':
                selected_mat = aggregate_parents(base, idx, f['outer_edges'], choice['mix'])
                selected_idx = idx
            else:
                log(f"Fold {i}/4 OUTER: selected {choice['name']} ({selected_config['dim']}d, {selected_config['epochs']} epochs)")
                selected_idx, embedded, _ = fit_native_config(f['outer_path'], SEED+i*11+1,
                                                                 selected_config)
                if need-set(selected_idx):
                    raise RuntimeError('Selected model missing heldout/control classes')
                selected_mat = aggregate_parents(embedded, selected_idx,
                                                  f['outer_edges'], choice['mix'])
            selected_stats = v9.stats(selected_mat, selected_idx, holdout, cmap)
            native_results.append(native_stats)
            chosen_results.append(selected_stats)
            fold_metrics.append({'fold': i, 'selected_setting': choice,
                                 'reference': {k:v for k,v in native_stats.items() if k!='rows'},
                                 'selected': {k:v for k,v in selected_stats.items() if k!='rows'},
                                 'train_file_sha256': f['outer_sha256'],
                                 'native_library_versions': versions})
            for baseline_row, candidate_row in zip(native_stats['rows'], selected_stats['rows']):
                if ((baseline_row['child'],baseline_row['expected_parent']) !=
                    (candidate_row['child'],candidate_row['expected_parent'])):
                    raise RuntimeError('Outer holdout identity mismatch')
                all_rows.append({'fold': i, 'record_type': 'HELDOUT',
                                 'child': baseline_row['child'],
                                 'expected_parent': baseline_row['expected_parent'],
                                 'native_cosine': baseline_row['cosine'],
                                 'v11_cosine': candidate_row['cosine'],
                                 'native_rank': baseline_row['rank_among_four'],
                                 'v11_rank': candidate_row['rank_among_four'],
                                 'native_control_mean':baseline_row['control_mean'],
                                 'v11_control_mean':candidate_row['control_mean']})
                for (pa, nv), (pb, cv) in zip(baseline_row['controls'], candidate_row['controls']):
                    if pa!=pb: raise RuntimeError('Control parent identity mismatch')
                    all_rows.append({'fold': i,'record_type':'UNASSERTED_PARENT_CONTROL',
                                     'child': baseline_row['child'], 'control_parent': pa,
                                     'native_cosine': nv, 'v11_cosine': cv})
            log(f"Fold {i}: native cosine={native_stats['positive_mean']:.4f}; "
                f"v11 cosine={selected_stats['positive_mean']:.4f}; "
                f"v11 AUC={selected_stats['auc']:.3f}")
            v8.save_json(report/f'fold_{i}_metrics.json', fold_metrics[-1])

        native = pool_stats(native_results)
        chosen = pool_stats(chosen_results)
        baseline = reference['v9_metric_summary']
        delta = chosen['positive_mean']-native['positive_mean']
        gates = {'cosine_at_least_0_70': chosen['positive_mean'] >= .70,
                 'auc_close_to_matched_native': chosen['auc'] >= native['auc']-.02,
                 'control_gap_close_to_matched_native': chosen['gap'] >= native['gap']-.04,
                 'top1_at_least_native_minus_one': chosen['top1_of_four'] >= native['top1_of_four']-1}
        final_status = ('TARGET_AND_CONTROLS_MET_EXPLORATORY'
                        if all(gates.values()) else 'TARGET_OR_CONTROLS_NOT_MET')
        v8.save_csv(report/'all_pair_scores.csv', all_rows,
                    ['fold','record_type','child','expected_parent','control_parent',
                     'native_cosine','v11_cosine','native_rank','v11_rank',
                     'native_control_mean','v11_control_mean'])
        output = {
            'version': VERSION, 'status': final_status,
            'selection_source':'inner_source_import_only_no_rail_target',
            'outer_rail_targets_used_to_tune':False,
            'native_configurations_compared':len(NATIVE_CONFIGS),
            'one_hop_mixes_compared': list(MIXES),
            'selected_global_setting':choice,
            'native_mean_cosine':native['positive_mean'],
            'v11_mean_cosine':chosen['positive_mean'],
            'delta_mean_cosine':delta,
            'v9_frozen_mean_cosine':baseline['native_8_class_mean'],
            'delta_from_frozen_v9':chosen['positive_mean']-baseline['native_8_class_mean'],
            'native_auc':native['auc'], 'v11_auc':chosen['auc'],
            'native_gap':native['gap'], 'v11_gap':chosen['gap'],
            'native_top1_of_four':native['top1_of_four'],
            'v11_top1_of_four':chosen['top1_of_four'],
            'native_control_mean':native['control_mean'],
            'v11_control_mean':chosen['control_mean'],
            'frozen_v9_reference':baseline, 'metrics_gates':gates,
            'development_baseline':{k:base_dev[k] for k in
                ('positive_mean','auc','gap','top1_of_four')},
            'data':{'folds':4, 'heldout_rail_pairs':8,
                    'comparison_parent_pairs':24,'development_pairs':160,
                    'approved_superclass_anchors':7,
                    'training_owl_el_normal_form':'GCI0_ONLY'},
            'provenance':{'input_sha256':snapshot,
                          'review_file_sha256':v8.digest(review),
                          'v9_reference_file_sha256':reference['reference_files_sha256'],
                          'source_or_pipeline_modified':False},
            'named_subclass_leakage_detected':False,
            'named_only_hermit_previously_passed':True,
            'complete_owl_import_closure_verified':False,
            'semantics_assumed_valid_as_user_requested':True,
            'exploratory_reused_rail_test_set':True,
            'negative_alternatives_are_unasserted_not_owl_disjoint':True,
        }
        if v8.immutability_snapshot(root)!=snapshot:
            raise RuntimeError('Upstream ontology or pipeline file altered')
        v10.check_reference_unchanged(root, reference)
        v8.save_json(report/'eight_class_summary.json', output)
        log('Frozen v9 cosine: '+str(baseline['native_8_class_mean']))
        log('Matched native cosine: '+str(native['positive_mean']))
        log('v11 held-out cosine: '+str(chosen['positive_mean']))
        log('v11 delta vs matched native: '+('%+.6f'%delta))
        log('v11 AUC: '+str(chosen['auc'])+'; top-1: '+str(chosen['top1_of_four'])+'/8')
        log('Result: '+final_status)
        log('Report: reports/experiments/'+TAG+'/eight_class_summary.json')
        return 0
    except Exception as error:
        print('[Rail mOWL v11] FAILED: '+repr(error),file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
