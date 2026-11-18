"""Project 6: separate Project-5-derived joint mOWL+cosine experiment.

Uses the EXISTING Project-6 80/20 split. Tunes on an additional internal split
of training; evaluates outer validation just once. Does not edit run_all.py,
original ontology, original mOWL vectors, or any Step-5 summaries.

python -u scripts/experiment_joint_rail_mowl.py --project-root .
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL

VERSION = 'project6-joint-EL-cosine-project5-transfer-v1'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def rdf(path: Path) -> Graph:
    g = Graph()
    if path.suffix.lower() in {'.ttl', '.turtle'}:
        g.parse(str(path), format='turtle')
    elif path.suffix.lower() in {'.rdf', '.owl', '.xml'}:
        try:
            g.parse(str(path), format='xml')
        except Exception:
            g.parse(str(path), format='turtle')
    else:
        g.parse(str(path))
    return g


def named_pairs(g: Graph) -> set[tuple[str, str]]:
    return {(str(s), str(o)) for s, o in g.subject_objects(RDFS.subClassOf)
            if isinstance(s, URIRef) and isinstance(o, URIRef) and s != o}


def closure(edges: set[tuple[str,str]]) -> set[tuple[str,str]]:
    parents = defaultdict(set)
    for a, b in edges:
        parents[a].add(b)
    out = set()
    for start in sorted(parents):
        stack, seen = list(parents[start]), set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node != start:
                out.add((start, node))
            stack.extend(parents[node] - seen)
    return out


def reachable(edges, start, target):
    parents = defaultdict(set)
    for a, b in edges:
        parents[a].add(b)
    stack, seen = [start], set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(parents[node] - seen)
    return False


def pair_nodes(edges):
    return {x for pair in edges for x in pair}


def degrees(edges):
    counts = Counter()
    for a, b in edges:
        counts[a] += 1
        counts[b] += 1
    return counts


def internal_dev_split(all_train: set[tuple[str,str]], seed: int, fraction: float,
                       target_min: int = 3) -> tuple[set, set]:
    """Choose only edges whose endpoints remain represented and aren't implied."""
    if not 0.05 <= fraction <= 0.4:
        raise ValueError('Internal validation fraction must be 0.05..0.40')
    train, held = set(all_train), set()
    target = max(target_min, round(len(train) * fraction))
    choices = sorted(train)
    random.Random(seed).shuffle(choices)
    for edge in choices:
        if len(held) >= target:
            break
        proposed = train - {edge}
        support = degrees(proposed)
        if not support[edge[0]] or not support[edge[1]]:
            continue
        if reachable(proposed, *edge):
            continue
        train.remove(edge)
        held.add(edge)
    if len(held) < target_min:
        raise ValueError('Insufficient supported independent training edges for development split')
    if held & (train | closure(train)):
        raise AssertionError('Internal development edge leakage')
    return train, held


def check_inputs(root: Path) -> dict:
    s = root/'reports/step5_split_report.json'
    b = root/'reports/step5_cosine_scores.json'
    t = root/'generated/mowl_rail_train.ttl'
    v = root/'generated/mowl_rail_valid.ttl'
    src = root/'src/module.ttl'
    refined = root/'generated/rail_axioms_refined.ttl'
    for p in (s, b, t, v, src, refined):
        if not p.is_file():
            raise FileNotFoundError(f'Expected fresh Project 6 input: {p}')
    split, baseline = read_json(s), read_json(b)
    for path, key, metadata in [
        (t, 'train_sha256', split), (v, 'valid_sha256', split),
        (src, 'source_sha256', split), (refined, 'refined_sha256', split),
        (t, 'train_sha256', baseline), (v, 'valid_sha256', baseline)]:
        if digest(path) != metadata.get(key):
            raise ValueError(f'Provenance mismatch: {path} ({key})')
    if split.get('rail_candidate_holdout_pairs', 0) < 2:
        raise ValueError('Install/execute 80/20 Step 5A first; at least two held-out rail pairs required')
    if abs(split.get('validation_fraction_requested', -1) - 0.2) > .000001:
        raise ValueError('This experiment requires the previous 80/20 outer split')
    if baseline.get('heldout_rail_pairs') != split['rail_candidate_holdout_pairs']:
        raise ValueError('Baseline was not produced from current 80/20 split')
    train, valid = named_pairs(rdf(t)), named_pairs(rdf(v))
    required_train = split['train_axiom_pairs']
    required_valid = split['valid_axiom_pairs']
    if len(train) != required_train or len(valid) != required_valid:
        raise ValueError('Train/validation report counts disagree with RDF')
    hold = {(r['child'], r['parent']) for r in split['holdout_pairs']}
    if valid != hold or train & valid or valid & closure(train):
        raise ValueError('Direct or transitive train/validation leakage')
    candidate = {(r['child'],r['parent']) for r in split['candidate_pairs']}
    rail_hold = {x for x in valid if x in candidate}
    base_hold = valid - rail_hold
    if len(rail_hold) != split['rail_candidate_holdout_pairs'] or len(base_hold) != split['source_import_holdout_pairs']:
        raise ValueError('Current evaluation partitions do not match 80/20 report')
    return dict(split=split, baseline=baseline, train=train, valid=valid,
                candidate=candidate, rail_hold=rail_hold, base_hold=base_hold,
                path_train=t, path_valid=v, source=src, refined=refined)


def collect_text(root: Path, inputs: dict) -> dict[str,str]:
    """Only annotations, never heldout subclass triples, enter the text features.

    All files are provenance checked before use, including local imports.
    Class definitions/labels are not axioms; semantic descriptions might still
    implicitly disclose taxonomy hints, so report this caveat.
    """
    paths = [inputs['source'], inputs['refined']]
    for rel, sha in sorted(inputs['split'].get('import_hashes',{}).items()):
        p = root/rel
        if not p.is_file() or digest(p) != sha:
            raise ValueError(f'Import changed since Step 5A: {rel}')
        paths.append(p)
    texts = defaultdict(list)
    for path in paths:
        graph = rdf(path)
        for s, p, o in graph:
            if not isinstance(s, URIRef) or not isinstance(o, Literal):
                continue
            name = str(p).rsplit('/',1)[-1].rsplit('#',1)[-1].lower()
            if name in ('label','preflabel') or ('definition' in name or 'description' in name or 'comment' in name or 'elucidation' in name):
                raw = str(o).strip()
                if raw:
                    texts[str(s)].append(raw)
    return {k:' '.join(sorted(set(values))) for k,values in texts.items()}


def hashed_text_features(index: dict[str,int], texts: dict[str,str], width=256):
    """Project 5's deterministic label/definition semantic adapter input."""
    if width < 8:
        raise ValueError('Text feature width must be >=8')
    docs = {}
    for iri in index:
        tail = iri.rsplit('/',1)[-1].rsplit('#',1)[-1]
        tail = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ',tail).replace('_',' ')
        docs[iri] = re.findall(r'[a-z][a-z0-9]{1,}', (texts.get(iri,'')+' '+tail).lower())
    counts = Counter()
    for tokens in docs.values():
        counts.update(set(tokens))
    out = torch.zeros((len(index),width),dtype=torch.float32)
    for iri,idx in index.items():
        for token, count in Counter(docs[iri]).items():
            slot = int.from_bytes(hashlib.blake2b(token.encode(),digest_size=8).digest(),'big') % width
            out[idx,slot] += math.log1p(count)*(1+math.log((1+len(index))/(1+counts[token])))
    return F.normalize(out,p=2,dim=1)


def unasserted_controls(available: list[str], forbidden: set[tuple[str,str]], n: int, seed:int) -> list[tuple[str,str]]:
    """Unasserted structural controls, NEVER proven OWL falsehoods.

    Known outer validation pairs are excluded solely from control generation,
    never included in positive training, loss, tuning or checkpoint selection.
    """
    if len(available) < 3:
        return []
    rng = random.Random(seed)
    controls = set()
    for _ in range(max(1,n*80)):
        if len(controls) >= n:
            break
        a,b = rng.sample(available, 2)
        if (a,b) not in forbidden:
            controls.add((a,b))
    return sorted(controls)



def matched_child_controls(index: dict[str,int], positives: set[tuple[str,str]],
                           forbidden: set[tuple[str,str]], per_positive: int, seed: int) -> list[tuple[str,str]]:
    """Harder unasserted controls sharing each held-out child's IRI."""
    nodes = sorted(index)
    rng = random.Random(seed)
    selected=set()
    for child, parent in sorted(positives):
        if child not in index or parent not in index:
            raise ValueError('Missing rail child or parent vector')
        choices=[x for x in nodes if x!=child and x!=parent and (child,x) not in forbidden]
        if len(choices)<per_positive:
            raise ValueError(f'Insufficient legal controls for {child}')
        rng.shuffle(choices)
        for other in choices[:per_positive]:
            selected.add((child,other))
    return sorted(selected)


def pair_index(pairs, index):
    missing = pair_nodes(pairs) - set(index)
    if missing:
        raise ValueError(f'Classes missing from mOWL index: {sorted(missing)[:5]}')
    return torch.tensor([(index[a],index[b]) for a,b in pairs],dtype=torch.long)


def similarity_rows(vectors: torch.Tensor, edges, index, scope):
    if not edges:
        return []
    ordered=sorted(edges)
    inds=pair_index(ordered,index)
    with torch.no_grad():
        norm = F.normalize(vectors,p=2,dim=1)
        val = (norm[inds[:,0]]*norm[inds[:,1]]).sum(dim=1).cpu().tolist()
    return [dict(child=a,parent=b,cosine=round(float(c),6),scope=scope) for (a,b),c in zip(ordered,val)]


def stats(rows):
    vals=[x['cosine'] for x in rows]
    return {'count':len(vals),'mean':round(float(np.mean(vals)),6) if vals else None,
            'minimum':min(vals) if vals else None,'maximum':max(vals) if vals else None,
            'at_or_above_0_70':sum(v>=.70 for v in vals)}


def auc_pos_control(positives, controls):
    if not positives or not controls:
        return None
    yes=np.asarray([p['cosine'] for p in positives]); no=np.asarray([n['cosine'] for n in controls])
    return round(float(np.mean((yes[:,None]>no[None,:]).astype(float)+.5*(yes[:,None]==no[None,:]).astype(float))),6)


def model_matrix(model, text_features, semantic_scale):
    centers=model.module.class_embed.weight
    adapted=F.normalize(model.module.semantic_adapter(text_features.to(centers.device)),dim=1)
    return F.normalize(centers+semantic_scale*adapted,dim=1)


def build_joint(dataset, dim, features, positives, controls, cfg):
    from mowl.models import ELEmbeddings
    class JointEL(ELEmbeddings):
        def get_regularization_loss(self):
            native=super().get_regularization_loss()
            emb=model_matrix(self,self.features,self.semantic_scale)
            p=self.pos_ids
            sim=(emb[p[:,0]]*emb[p[:,1]]).sum(dim=1)
            value=native + self.lambda_cos*(1.0-sim).mean()
            if self.dispersion_weight and self.control_ids is not None:
                c=self.control_ids
                non=(emb[c[:,0]]*emb[c[:,1]]).sum(dim=1)
                value = value+self.dispersion_weight*F.relu(non-0.3).square().mean()
            return value
    model=JointEL(dataset,embed_dim=dim,margin=.1,reg_norm=1.,
                  learning_rate=cfg['learning_rate'],batch_size=128,
                  model_filepath=None,device='cpu')
    model.module.semantic_adapter=nn.Linear(features.shape[1],dim,bias=False)
    nn.init.xavier_uniform_(model.module.semantic_adapter.weight,gain=.1)
    model.features=features
    model.pos_ids=positives
    model.control_ids=controls
    model.semantic_scale=cfg['semantic_scale']
    model.lambda_cos=cfg['lambda_cos']
    model.dispersion_weight=cfg['dispersion_weight']
    return model


def write_training_graph(original: Graph, allowed_edges: set[tuple[str,str]], destination: Path):
    g=Graph()
    for s,o in original.subject_objects(RDF.type):
        if o==OWL.Class:
            g.add((s,RDF.type,OWL.Class))
    for a,b in sorted(allowed_edges):
        g.add((URIRef(a),RDFS.subClassOf,URIRef(b)))
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix('.ttl.tmp')
    g.serialize(destination=str(tmp),format='turtle')
    tmp.replace(destination)
    return g


def train_once(root, graph_path, train_edges, eval_edges, forbidden, texts, cfg,
               epochs, seed, max_inferred, control_count, text_width, stage):
    """Native ELEmbeddings loss + P5 cosine/semantic auxiliary regularization."""
    from mowl.datasets import PathDataset, Dataset
    from mowl.models import ELEmbeddings
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ds=Dataset(PathDataset(str(graph_path.resolve())).ontology)
    base=ELEmbeddings(ds,embed_dim=cfg['dimension'],margin=.1,reg_norm=1.0,
                      learning_rate=cfg['learning_rate'],batch_size=128,model_filepath=None,device='cpu')
    index={str(iri):int(idx) for iri,idx in base.class_index_dict.items()}
    del base
    if pair_nodes(train_edges | eval_edges) - set(index):
        raise ValueError('Class declarations/indexing incomplete; cannot validate')
    inferred=closure(train_edges)-train_edges
    if eval_edges & (train_edges | inferred):
        raise ValueError('Training closure entails validation; abort')
    inferred=sorted(inferred)[:max_inferred]
    positive=sorted(train_edges | set(inferred))
    positives_idx=pair_index(positive,index)
    # Mask all known positives (including outer heldouts) so they are not
    # incorrectly treated as negative controls. Their SCORES are not used here.
    blocked=set(forbidden)|set(train_edges)|set(inferred)|closure(train_edges)
    controls=unasserted_controls(sorted(index),blocked,control_count,seed)
    controls_idx=pair_index(controls,index) if controls else None
    features=hashed_text_features(index,texts,width=text_width)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    joint=build_joint(ds,cfg['dimension'],features,positives_idx,controls_idx,cfg)
    if {str(k):int(v) for k,v in joint.class_index_dict.items()} != index:
        raise RuntimeError('mOWL class index changed between baseline and joint model')
    before=joint.module.semantic_adapter.weight.detach().clone()
    joint.train(epochs=epochs,validate_every=epochs)
    with torch.no_grad():
        matrix=model_matrix(joint,features,cfg['semantic_scale']).detach().clone()
    adapter_changed=bool((joint.module.semantic_adapter.weight.detach()-before).abs().max().item()>1e-9)
    if not adapter_changed:
        raise RuntimeError('Semantic adapter did not receive optimization updates')
    evaluation=similarity_rows(matrix,eval_edges,index,stage)
    controls_scores=similarity_rows(matrix,controls,index,stage+'_unasserted_controls')
    estats=stats(evaluation); cstats=stats(controls_scores)
    separation=estats['mean']-cstats['mean'] if cstats['mean'] is not None else None
    outcome={'stage':stage,'config':cfg,'epochs':epochs,'mean_cosine':estats['mean'],
             'control_mean':cstats['mean'],'positive_control_gap':round(separation,6) if separation is not None else None,
             'positive_control_auc':auc_pos_control(evaluation,controls_scores),
             'inferred_training_pairs':len(inferred),'total_training_positive_pairs':len(positive),
             'semantic_adapter_updated':adapter_changed,'score_count':len(evaluation),
             'train_positive_mean_cosine':stats(similarity_rows(matrix,train_edges,index,stage+'_train'))['mean']}
    return outcome,evaluation,controls_scores,matrix,index


def select_score(held_mean, gap):
    """Project 5 rule: tune on inner dev only, discourage collapsed controls."""
    if held_mean is None or gap is None:
        return -1e9
    return round(float(held_mean-.3*max(0.,-gap)),6)


def run(root:Path, *, seed=42, tune_epochs=250, final_epochs=700,
        configs=8, max_inferred=600, controls=120, dev_fraction=.16,
        text_width=256, threshold=.70) -> dict:
    root=root.resolve()
    if not (0<threshold<=1) or min(tune_epochs,final_epochs,configs,max_inferred,controls,text_width)<1:
        raise ValueError('Invalid training parameters')
    inputs=check_inputs(root)
    original_train=rdf(inputs['path_train'])
    dev_train,dev_hold=internal_dev_split(inputs['train'],seed,dev_fraction)
    dev_graph_path=root/'generated/mowl_joint_project5_dev_train.ttl'
    write_training_graph(original_train,dev_train,dev_graph_path)
    if named_pairs(rdf(dev_graph_path)) != dev_train:
        raise ValueError('Development training projection mismatch')
    texts=collect_text(root,inputs)
    import mowl
    mowl.init_jvm('2g')
    configurations=[dict(dimension=dim,lambda_cos=lam,semantic_scale=scale,
                         learning_rate=.001,dispersion_weight=disp)
                    for dim,lam,scale,disp in itertools.product(
                        (32,50),(.25,.75,1.5),(.15,.4),(0.,.02))]
    random.Random(seed).shuffle(configurations)
    configurations=configurations[:configs]
    sweep=[]; selected=None
    dev_forbidden=set(inputs['valid'])|set(dev_hold)|set(inputs['train'])
    for i,cfg in enumerate(configurations,1):
        print(f'[Joint mOWL] Inner development config {i}/{len(configurations)}: {cfg}',flush=True)
        outcome,_,_,_,_=train_once(root,dev_graph_path,dev_train,dev_hold,
                                    dev_forbidden,texts,cfg,tune_epochs,seed,max_inferred,
                                    controls,text_width,f'development_{i}')
        outcome['selection_score']=select_score(outcome['mean_cosine'],outcome['positive_control_gap'])
        sweep.append(outcome)
        if selected is None or outcome['selection_score']>selected['selection_score']:
            selected=outcome
    cfg=selected['config']
    print(f'[Joint mOWL] Selected using INTERNAL development only: {cfg}',flush=True)
    outer_forbidden=set(inputs['valid'])|set(inputs['train'])
    outcome,all_rows,control_rows,mat,index=train_once(
        root,inputs['path_train'],inputs['train'],inputs['valid'],outer_forbidden,
        texts,cfg,final_epochs,seed,max_inferred,controls,text_width,'outer_final')
    rail_rows=[r for r in all_rows if (r['child'],r['parent']) in inputs['rail_hold']]
    base_rows=[r for r in all_rows if (r['child'],r['parent']) in inputs['base_hold']]
    rail_stat,base_stat,all_stat=stats(rail_rows),stats(base_rows),stats(all_rows)
    control_stat=stats(control_rows)
    # A deliberately harder rail-specific diagnostic: replace the parent
    # with an unasserted alternative while keeping the SAME rail child.
    # No holdout examples enter training through these diagnostics.
    blocked=set(inputs['train'])|set(inputs['valid'])|closure(inputs['train'])
    rail_controls=matched_child_controls(index,inputs['rail_hold'],blocked,3,seed+101)
    rail_controls_rows=similarity_rows(mat,rail_controls,index,'outer_rail_matched_child_control')
    rail_controls_stat=stats(rail_controls_rows)
    rail_control_auc=auc_pos_control(rail_rows,rail_controls_rows)
    rail_gap=(rail_stat['mean']-rail_controls_stat['mean'] if rail_controls_stat['mean'] is not None else None)
    # No full-run success can be claimed unless BOTH domain and general held-out
    # averages satisfy the user-specified 0.70 rule.
    rail_pass=rail_stat['mean'] is not None and rail_stat['mean']>=threshold
    full_pass=all_stat['mean'] is not None and all_stat['mean']>=threshold
    gap=(all_stat['mean']-control_stat['mean'] if control_stat['mean'] is not None else None)
    auc=auc_pos_control(all_rows,control_rows)
    control_pass=(gap is not None and gap>=.15 and auc is not None and auc>=.70 and
                  rail_gap is not None and rail_gap>=.15 and rail_control_auc is not None and rail_control_auc>=.70)
    status=('PASS_HOLDOUT_COSINE_AND_CONTROL' if rail_pass and full_pass and control_pass
            else 'FAIL_HOLDOUT_COSINE_OR_DISCRIMINATION')
    # Final invariants; do not overwrite original ELEmbeddings vector artifact.
    for rel,sha in inputs['split'].get('import_hashes',{}).items():
        if digest(root/rel)!=sha:
            raise RuntimeError(f'Local import changed during experiment: {rel}')
    for path,key in [(inputs['source'],'source_sha256'),(inputs['refined'],'refined_sha256'),
                     (inputs['path_train'],'train_sha256'),(inputs['path_valid'],'valid_sha256')]:
        if digest(path)!=inputs['split'][key]:
            raise RuntimeError(f'Input modified during experiment: {path}')
    vector_path=root/'generated/mowl_joint_project5_class_vectors.npz'
    ordered=sorted(index)
    tmp=vector_path.with_name(vector_path.name+'.tmp')
    with tmp.open('wb') as fh:
        np.savez_compressed(fh,iris=np.asarray(ordered,dtype='U'),
                            vectors=np.asarray([mat[index[iri]].cpu().numpy() for iri in ordered],dtype=np.float32))
    tmp.replace(vector_path)
    report={'version':VERSION,'status':status,'threshold':threshold,
            'threshold_met':bool(rail_pass and full_pass and control_pass),
            'rail_threshold_met':bool(rail_pass),'overall_threshold_met':bool(full_pass),
            'control_guard_met':bool(control_pass),
            'rail_holdout':rail_stat,'source_import_holdout':base_stat,
            'all_holdout':all_stat,'unasserted_control':control_stat,
            'all_holdout_minus_control':round(gap,6) if gap is not None else None,
            'all_holdout_control_auc':auc,
            'rail_matched_child_control':rail_controls_stat,
            'rail_matched_child_control_auc':rail_control_auc,
            'rail_positive_minus_matched_child_control':round(rail_gap,6) if rail_gap is not None else None,
            'rail_matched_child_control_pairs':rail_controls_rows,
            'heldout_rail_pairs':rail_rows,'heldout_all_pairs':all_rows,
            'unasserted_control_pairs':control_rows,
            'development':{'train_pairs':len(dev_train),'heldout_pairs':len(dev_hold),
                           'chosen_config':cfg,'chosen_development':selected,
                           'sweep':sweep,'outer_validation_not_used_for_selection':True},
            'final_fit':outcome,
            'input_sha256':{'source':inputs['split']['source_sha256'],
                            'refined':inputs['split']['refined_sha256'],
                            'train':inputs['split']['train_sha256'],
                            'valid':inputs['split']['valid_sha256']},
            'model':{'architecture':'native mOWL ELEmbeddings + joint cosine regularizer + hashed label/definition adapter',
                     'vector_file':str(vector_path.relative_to(root)),
                     'vector_sha256':digest(vector_path),'seed':seed,
                     'tuning_epochs':tune_epochs,'final_epochs':final_epochs,
                     'mowl_version':getattr(mowl,'__version__','unknown'),
                     'torch_version':torch.__version__},
            'baseline_mowl':{
                'heldout_rail_mean_cosine':inputs['baseline']['mean_heldout_rail_cosine'],
                'heldout_source_import_mean_cosine':inputs['baseline']['mean_heldout_source_import_cosine'],
                'baseline_version':inputs['baseline']['version']},
            'leakage_audit':{'direct_or_transitive_outer':False,
                             'direct_or_transitive_development':False,
                             'all_training_inferred_pairs_from_train_only':True,
                             'outer_heldout_pairs_used_as_training_positives':False,
                             'outer_validation_used_for_hyperparameter_selection':False},
            'limitations':['Only taxonomy named subclass + class declarations used for native mOWL loss; training-only inferred closure is added to cosine alignment, not OWL asserted input.',
                           'Two held-out rail pairs are a small sample.',
                           'Text features come from existing class labels and definitions, which may contain implicit taxonomy hints.',
                           'Unasserted control pairs are not verified OWL-negative statements.',
                           'High raw center-vector cosine does not establish OWL logical validity.',
                           'Do not call this model Project 6 main pipeline performance until experimentally verified and integrated.']}
    report_path=root/'reports/step5_joint_project5_comparison.json'
    write_json(report_path,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tune-epochs',type=int,default=250)
    p.add_argument('--final-epochs',type=int,default=700)
    p.add_argument('--configs',type=int,default=8)
    p.add_argument('--max-inferred',type=int,default=600)
    p.add_argument('--controls',type=int,default=120)
    p.add_argument('--dev-fraction',type=float,default=.16)
    p.add_argument('--text-width',type=int,default=256)
    p.add_argument('--threshold',type=float,default=.70)
    p.add_argument('--strict',action='store_true',help='Exit nonzero if heldout threshold/control guard fails')
    p.add_argument('--preflight-only',action='store_true',help='Check 80/20 and imports without training mOWL')
    a=p.parse_args()
    try:
        if a.preflight_only:
            inp=check_inputs(a.project_root.resolve())
            dtrain,dhold=internal_dev_split(inp['train'],a.seed,a.dev_fraction)
            texts=collect_text(a.project_root.resolve(),inp)
            print(json.dumps({'status':'PASS_JOINT_MOWL_EXPERIMENT_PREFLIGHT',
                 'outer_train_pairs':len(inp['train']),
                 'outer_validation_pairs':len(inp['valid']),
                 'outer_rail_holdouts':len(inp['rail_hold']),
                 'development_train_pairs':len(dtrain),
                 'development_holdout_pairs':len(dhold),
                 'annotated_classes':len(texts)},indent=2))
            return 0
        r=run(a.project_root,seed=a.seed,tune_epochs=a.tune_epochs,
              final_epochs=a.final_epochs,configs=a.configs,max_inferred=a.max_inferred,
              controls=a.controls,dev_fraction=a.dev_fraction,text_width=a.text_width,
              threshold=a.threshold)
    except Exception as exc:
        print(f'JOINT MOWL EXPERIMENT - ERROR: {exc}',file=sys.stderr)
        return 2
    print('JOINT MOWL EXPERIMENT - '+r['status'])
    print(json.dumps({'rail':r['rail_holdout'],'source_import':r['source_import_holdout'],
                      'all':r['all_holdout'],'controls':r['unasserted_control'],
                      'auc':r['all_holdout_control_auc'],
                      'threshold_met':r['threshold_met'],
                      'report':'reports/step5_joint_project5_comparison.json'},indent=2))
    return 1 if a.strict and not r['threshold_met'] else 0

if __name__=='__main__':
    raise SystemExit(main())
