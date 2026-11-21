"""Isolated Project 6 experiment: Project 5 v8 frozen ELEmbeddings + residual MLP.

From projects/project-6 root:
  python -u scripts/experiment_rail_mowl_v8.py --prepare-only
  python -u scripts/experiment_rail_mowl_v8.py

The four rail folds are the eight primary superclass assertions in the
rail_holdout_feasibility.json audit. Each fold holds out TWO rail edges;
none of those edges is used in that fold's mOWL or projection training.

IMPORTANT: A class without retained training relationships is cold-start.

Optional --reviewed-anchors permits exactly the seven human-approved broader
parent anchors. It validates them, rebuilds per-fold training files in isolated
paths, and does NOT train on any of the eight target subclass assertions.

Human approval is not an OWL reasoner proof; the prepared graph is explicitly
restricted to named subclass axioms and declarations. A full OWL entailment
check is still required before claiming OWL-level validation.
Cosine results for such classes do not establish learned ontology alignment.
Incorrect-parent controls are *unasserted alternatives*, not OWL negatives.

Inputs (all read-only):
  src/module.ttl
  generated/module_augmented.ttl
  generated/mowl_rail_train.ttl
  generated/mowl_rail_valid.ttl
  reports/step5_split_report.json
  imports/ (if present)

All written artifacts are isolated beneath:
  generated/experiments/rail_mowl_v8/
  reports/experiments/rail_mowl_v8/

Does not import or modify Project 6 run_all.py or upstream stages.
Requires rdflib, numpy, torch, mowl, JVM (in the existing .venv-mowl).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

VERSION = 'project6-isolated-project5-v8-rail-four-fold-v1'
BASE = 'https://example.org/ontologies/intel-ontology#'
AGENT = 'https://www.commoncoreontologies.org/AgentOntology/'
BFO = 'http://purl.obolibrary.org/obo/'

# Audited rails: four independent two-edge tests, not a random resplit.
FOLDS: tuple[tuple[tuple[str, str], ...], ...] = (
    ((BASE+'rail_passenger_role', AGENT+'passenger'),
     (BASE+'rail_reservation_information', AGENT+'reservation')),
    ((BASE+'railway_station', BFO+'BFO_0000040'),
     (BASE+'train', BFO+'BFO_0000040')),
    ((BASE+'rail_journey_plan', BFO+'IAO_0000030'),
     (BASE+'rail_ticket', AGENT+'ticket')),
    ((BASE+'passenger_train', BASE+'train'),
     (BASE+'rail_journey', BFO+'BFO_0000015')),
)


# The EXACT seven broader subclass axioms approved by the Project 6 ontology
# author for experimental use, 2026-11-19. Do not replace with automatically
# discovered ancestors or accept additional edges in this experiment.
APPROVED_SEVEN: frozenset[tuple[str,str]] = frozenset({
    (BASE+'rail_reservation_information', BFO+'BFO_0000031'),
    (BASE+'railway_station', BFO+'BFO_0000004'),
    (BASE+'train', BFO+'BFO_0000004'),
    (BASE+'rail_journey_plan', BFO+'BFO_0000031'),
    (BASE+'rail_ticket', BFO+'IAO_0000178'),
    (BASE+'passenger_train', BFO+'BFO_0000040'),
    (BASE+'rail_journey', BFO+'BFO_0000003'),
})
REVIEWED_RUN_ID = 'rail_mowl_v8_reviewed'

# Fixed *a priori*: do NOT tune or select on these eight holdout examples.
PROJECTION = dict(learning_rate=0.0025, epochs=100, hidden_size=32,
                  identity_weight=0.10, residual_scale=0.25,
                  weight_decay=1e-5, positive_weight=1.0,
                  residual_penalty_weight=0.001)
EMBED_DIM = 50
DEFAULT_NATIVE_EPOCHS = 1000
TARGET = 0.70


def log(message: str) -> None:
    print(f'[Rail mOWL v8] {message}', flush=True)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def mandatory_inputs(root: Path) -> list[Path]:
    paths = [root/'src/module.ttl', root/'generated/module_augmented.ttl',
             root/'generated/mowl_rail_train.ttl', root/'generated/mowl_rail_valid.ttl',
             root/'reports/step5_split_report.json']
    absent = [str(p) for p in paths if not p.is_file()]
    if absent:
        raise FileNotFoundError('Run upstream Project 6 stages first; missing READ-ONLY input(s):\n'+'\n'.join(absent))
    return paths


def immutability_snapshot(root: Path) -> dict[str, str]:
    paths = mandatory_inputs(root)
    pipeline = root/'scripts/run_all.py'
    if pipeline.is_file():
        paths.append(pipeline)
    if (root/'imports').exists():
        paths.extend(sorted(p for p in (root/'imports').rglob('*') if p.is_file()))
    return {p.relative_to(root).as_posix(): digest(p) for p in paths}


def graph_load(path: Path) -> Graph:
    graph = Graph()
    graph.parse(path, format='turtle')
    return graph


def named_pairs(g: Graph) -> set[tuple[str,str]]:
    return {(str(s),str(o)) for s,o in g.subject_objects(RDFS.subClassOf)
            if isinstance(s,URIRef) and isinstance(o,URIRef) and s!=o}


def closure(edges: set[tuple[str,str]]) -> set[tuple[str,str]]:
    parents: dict[str,set[str]] = defaultdict(set)
    for child,parent in edges:
        parents[child].add(parent)
    nodes = set(parents)
    for values in parents.values():
        nodes.update(values)
    reached = set()
    for child in nodes:
        queue = deque(parents.get(child,set()))
        seen = {child}
        while queue:
            ancestor=queue.popleft()
            if ancestor in seen: continue
            seen.add(ancestor)
            reached.add((child,ancestor))
            queue.extend(parents.get(ancestor,set())-seen)
    return reached


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    temp.replace(path)


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def save_training_graph(path: Path, edges: set[tuple[str,str]], classes: set[str]) -> None:
    # N-Triples is an RDF subset of Turtle. Sorted lines ensure deterministic bytes,
    # and mOWL/OWLAPI can parse *.ttl via PathDataset.
    lines=[f'<{iri}> <{RDF.type}> <{OWL.Class}> .' for iri in sorted(classes)]
    lines += [f'<{c}> <{RDFS.subClassOf}> <{p}> .' for c,p in sorted(edges)]
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    temp.replace(path)
    check=graph_load(path)
    if named_pairs(check)!=edges:
        raise RuntimeError('Serialized fold training set failed RDF round-trip')


def determine_support(edges: set[tuple[str,str]], child: str) -> dict:
    outgoing=sum(a==child for a,_ in edges)
    incoming=sum(b==child for _,b in edges)
    status = ('RETAINED_PARENT_SUPPORT' if outgoing else
              'INCOMING_ONLY_NO_PARENT_ANCHOR' if incoming else
              'COLD_START_NO_TAXONOMY_RELATION')
    return {'training_outgoing':outgoing,'training_incoming':incoming,
            'support_status':status}


def load_split(root: Path) -> dict:
    """Create new CV folds from the complete *existing* named taxonomy.

    Step5A's old source/import split is deliberately recombined here since it
    served a DIFFERENT validation objective. Only the eight rail assertions
    are used for evaluation in this new four-fold experiment.
    """
    report=json.loads((root/'reports/step5_split_report.json').read_text(encoding='utf-8'))
    if report.get('status')!='PASS_SPLIT_READY_FOR_MOWL':
        raise ValueError('Step 5A split report is not PASS_SPLIT_READY_FOR_MOWL')
    for field,relative in [('source_sha256','src/module.ttl'),
                           ('train_sha256','generated/mowl_rail_train.ttl'),
                           ('valid_sha256','generated/mowl_rail_valid.ttl')]:
        if report.get(field)!=digest(root/relative):
            raise RuntimeError(f'Provenance hash mismatch: {relative}')
    train=graph_load(root/'generated/mowl_rail_train.ttl')
    valid=graph_load(root/'generated/mowl_rail_valid.ttl')
    augmented=graph_load(root/'generated/module_augmented.ttl')
    source_training=named_pairs(train)
    source_validation=named_pairs(valid)
    existing=source_training|source_validation
    aug_pairs=named_pairs(augmented)
    expected={pair for fold in FOLDS for pair in fold}
    if len(expected)!=8 or len({child for child,_ in expected})!=8:
        raise AssertionError('Expected exactly 8 unique rail children')
    missing=expected-aug_pairs
    if missing:
        raise ValueError('Audited held-out rail assertions are missing from generated/module_augmented.ttl: '+repr(sorted(missing)))
    # ONLY rail-child subclass edges come from augmented. Source/import
    # taxonomy is supplied by the provenance-checked Step5A graphs.
    rail_child_iris={c for c,_ in expected}
    rail_edges={edge for edge in aug_pairs if edge[0] in rail_child_iris}
    full=existing|rail_edges
    nodes={str(s) for g in (train,valid,augmented)
           for s in g.subjects(RDF.type,OWL.Class) if isinstance(s,URIRef)}
    for child,parent in full:
        nodes.update((child,parent))
    if not nodes:
        raise ValueError('No ontology classes found in training data')
    # A feasibility audit, if present, must match these previously selected folds.
    audit=root/'reports/rail_holdout_feasibility.json'
    if audit.exists():
        data=json.loads(audit.read_text(encoding='utf-8'))
        actual={int(f['fold']):{(p['child'],p['parent']) for p in f['held_out_pairs']}
                for f in data.get('folds',[])}
        intended={i:set(fold) for i,fold in enumerate(FOLDS,1)}
        if actual!=intended:
            raise ValueError('rail_holdout_feasibility.json disagrees with the audited fold definitions')
    return {'edges':full,'classes':nodes,'expected':expected,'rail_edges':rail_edges,
            'source_pairs':len(existing),'rail_pairs':len(rail_edges),
            'old_step5a_validation_reused_as_training':len(source_validation),
            'audit_available':audit.is_file()}


def prepare(root: Path, inputs: dict, snapshot: dict[str,str]) -> tuple[list[dict], dict]:
    dest=root/'generated/experiments/rail_mowl_v8'
    edges=inputs['edges']
    expected=inputs['expected']
    folds=[]
    for number,holdout in enumerate(FOLDS,1):
        held=set(holdout)
        training=edges-held
        if held&training:
            raise AssertionError('Direct holdout leakage')
        if held&closure(training):
            raise RuntimeError(f'Fold {number}: transitive named subclass leakage')
        if not held.issubset(edges):
            raise RuntimeError(f'Fold {number}: holdout not present in full ontology')
        support={c:determine_support(training,c) for c,_ in holdout}
        path=dest/f'fold_{number}_train.ttl'
        save_training_graph(path,training,inputs['classes'])
        folds.append({'fold':number,'training_path':path,
                      'heldout':list(holdout),'training':training,
                      'support':support,'train_pairs':len(training),'train_sha256':digest(path)})
    manifest={'version':VERSION,'status':'PREPARED_NO_TRAINING',
              'fold_count':4,'heldout_classes':8,'two_heldout_per_fold':True,
              'source_import_pair_count':inputs['source_pairs'],
              'rail_child_subclass_pair_count':inputs['rail_pairs'],
              'source_import_step5a_validation_pairs_recombined_as_new_cv_training':inputs['old_step5a_validation_reused_as_training'],
              'input_sha256':snapshot,'feasibility_audit_found':inputs['audit_available'],
              'note':'Named-subclass-only projection. Retained labels and properties are NOT trainable OWL EL restrictions.',
              'folds':[{'fold':f['fold'],'train_pairs':f['train_pairs'],
                        'train_sha256':f['train_sha256'],
                        'heldout':[{'child':c,'parent':p,**f['support'][c]} for c,p in f['heldout']]}
                       for f in folds]}
    save_json(root/'reports/experiments/rail_mowl_v8/preparation.json',manifest)
    return folds,manifest



def prepare_approved(root:Path,inputs:dict,snapshot:dict[str,str],review_path:Path)->tuple[list[dict],dict,dict]:
    """Reuse the already-audited reviewed-fold writer; never alter v8 baseline.

    Parent approvals authorize independent edges, not the held-out targets.
    Model training is on simple named-subclass graphs only; therefore this
    function proves nonreachability in that named graph, NOT arbitrary imported
    OWL entailment. Review metadata is attached to every output.
    """
    import audit_rail_superclass_support as audit
    review_path=review_path.resolve()
    if not review_path.is_file():
        raise FileNotFoundError(f'Approved anchor review file not found: {review_path}')
    raw=json.loads(review_path.read_text(encoding='utf-8'))
    if raw.get('human_review_status')!='SEVEN_APPROVED_FOR_EXPERIMENT_2026_11_19':
        raise ValueError('Review file lacks the seven user approvals for this experiment')
    approval_keys={(item.get('child'),item.get('parent')) for item in raw.get('approved_anchors',[])}
    if approval_keys!=APPROVED_SEVEN or len(raw.get('approved_anchors',[]))!=7:
        raise ValueError('Exactly the seven user-approved anchors must be used; refusing additions, deletions or substitution')
    if raw.get('target_holdouts_must_remain_excluded') is not True:
        raise ValueError('Review file does not assert the target holdout policy')
    anchors=audit.validated_anchors(review_path,inputs)
    if set(anchors)!=APPROVED_SEVEN:
        raise RuntimeError('Validated anchors differ from seven approved axioms')
    if immutability_snapshot(root)!=snapshot:
        raise RuntimeError('Upstream inputs changed during approved-anchor validation')
    # The writer creates isolated reviewed datasets, with a round-trip parse and
    # a separate named reachability check for every fold.
    reviewed=audit.write_reviewed_folds(root,inputs,{},anchors,review_path,snapshot)
    graph_files=[]
    complete_edges=inputs['edges'] | set(anchors)
    for item in reviewed['folds']:
        idx=item['fold']
        path=root/item['training_path']
        held=list(FOLDS[idx-1])
        training=complete_edges-set(held)
        observed=named_pairs(graph_load(path))
        if observed!=training or item['sha256']!=digest(path):
            raise RuntimeError(f'Fold {idx}: reviewed training graph or manifest mismatched on reread')
        if any((c,p) in observed or (c,p) in closure(observed) for c,p in held):
            raise RuntimeError(f'Fold {idx}: target leakage detected on reread')
        support={c:determine_support(training,c) for c,_ in held}
        graph_files.append({'fold':idx,'training_path':path,'heldout':held,
                            'training':training,'support':support,
                            'train_pairs':len(training),'train_sha256':digest(path)})
    if len(graph_files)!=4 or any(len(f['heldout'])!=2 for f in graph_files):
        raise RuntimeError('Expected exactly four two-target validation folds')
    if any(sum(f['support'][c]['training_outgoing']==0 for c,_ in f['heldout']) for f in graph_files):
        raise RuntimeError('At least one rail class has no retained superclass after approval')
    prepared={'version':VERSION,'mode':'SEVEN_HUMAN_APPROVED_BROADER_ANCHORS',
              'status':'PREPARED_REVIEWED_NAMED_GRAPH_NO_TRAINING',
              'input_sha256':snapshot,'review_file_sha256':digest(review_path),
              'review_file':str(review_path),
              'approved_anchor_count':len(anchors),
              'approved_anchor_edges':[{'child':c,'parent':p} for c,p in anchors],
              'held_out_targets':[{'child':c,'parent':p} for f in FOLDS for c,p in f],
              'named_subclass_entailment_leakage_found':False,
              'named_only_training_graph':True,
              'full_owl_semantic_entailment_checked':False,
              'source_ontologies_modified':False,
              'caveat':('Approval is a model-design decision, not formal OWL validation. '
                        'Each training graph contains only class declarations and named subClassOf edges; '
                        'none of the 8 held-out targets is asserted or reachable by named subclass transitivity. '
                        'Full OWL reasoner validation of the richer ontologies remains outstanding.'),
              'folds':[{'fold':f['fold'],'train_pairs':f['train_pairs'],
                        'train_sha256':f['train_sha256'],
                        'training_path':f['training_path'].relative_to(root).as_posix(),
                        'heldout':[{'child':c,'parent':p,**f['support'][c]}
                                   for c,p in f['heldout']]}
                       for f in graph_files]}
    save_json(root/'reports/experiments'/REVIEWED_RUN_ID/'preparation.json',prepared)
    # Controls must avoid the NEW approved positive pairs too.
    for_change={**inputs,'edges':complete_edges,'reviewed_anchors':set(anchors)}
    if immutability_snapshot(root)!=snapshot:
        raise RuntimeError('Upstream inputs changed while creating approved folds')
    return graph_files,prepared,for_change


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_projection(dim:int, hidden:int, scale:float):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    class ResidualProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1=nn.Linear(dim,hidden)
            self.fc2=nn.Linear(hidden,dim)
            self.activation=nn.GELU()
            self.scale=scale
            nn.init.xavier_uniform_(self.fc1.weight,gain=0.1)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        def forward(self,x):
            residual=self.fc2(self.activation(self.fc1(x)))
            return functional.normalize(x+self.scale*residual,p=2,dim=1)
    return ResidualProjection()


def train_projection(base,index,training,config,seed):
    import torch
    import torch.nn.functional as functional
    seed_everything(seed)
    projector=make_projection(base.shape[1],config['hidden_size'],config['residual_scale']).to('cpu')
    optimizer=torch.optim.AdamW(projector.parameters(),lr=config['learning_rate'],
                                weight_decay=config['weight_decay'])
    positive=torch.tensor([(index[c],index[p]) for c,p in sorted(training)],dtype=torch.long)
    if not len(positive):
        raise ValueError('No positive training pairs')
    identity_ids=torch.unique(positive.reshape(-1))
    native= functional.normalize(base,dim=1,p=2)
    best_loss=math.inf
    best_state=None
    final={}
    for epoch in range(1,config['epochs']+1):
        projector.train()
        optimizer.zero_grad()
        out=projector(base)
        positive_sim=(out[positive[:,0]]*out[positive[:,1]]).sum(dim=1)
        positive_loss=(1-positive_sim).mean()
        identity_loss=(1-(out[identity_ids]*native[identity_ids]).sum(dim=1)).mean()
        res=projector.fc2(projector.activation(projector.fc1(base[identity_ids])))
        residual_loss=(res**2).mean()
        loss=(config['positive_weight']*positive_loss
              +config['identity_weight']*identity_loss
              +config['residual_penalty_weight']*residual_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(projector.parameters(),max_norm=5.0)
        optimizer.step()
        scalar=float(loss.detach().item())
        # v8-style checkpointing: training objective ONLY, never validation.
        if scalar<best_loss:
            best_loss=scalar
            best_state={key:t.detach().clone() for key,t in projector.state_dict().items()}
        final={'total_loss':scalar,'positive_loss':float(positive_loss.detach().item()),
               'identity_loss':float(identity_loss.detach().item()),
               'residual_loss':float(residual_loss.detach().item()),'epoch':epoch}
    if best_state is None:
        raise RuntimeError('Projection training did not produce a checkpoint')
    projector.load_state_dict(best_state)
    return projector,{**final,'best_training_loss':best_loss}


def score(matrix, pairs, index) -> list[float]:
    import torch
    import torch.nn.functional as functional
    normalized=functional.normalize(matrix,dim=1,p=2)
    with torch.no_grad():
        return [float(torch.dot(normalized[index[c]],normalized[index[p]]).item()) for c,p in pairs]


def rank_auc(pos:list[float],ctrl:list[float])->float|None:
    if not pos or not ctrl: return None
    return sum((1.0 if a>b else 0.5 if a==b else 0.0) for a in pos for b in ctrl)/(len(pos)*len(ctrl))


def controls(full: set[tuple[str,str]], holdout: tuple[tuple[str,str], ...], count:int=3) -> dict[str,list[str]]:
    # Alternative superclasses selected from the eight candidate parent classes.
    # Exclude anything reachable from that child in the FULL graph.
    from hashlib import sha256
    actual_ancestors=closure(full)
    options=sorted({p for fold in FOLDS for _,p in fold})
    result={}
    for child,parent in holdout:
        pool=[p for p in options if p!=parent and p!=child and (child,p) not in actual_ancestors]
        if len(pool)<count:
            raise ValueError(f'Insufficient unentailed named controls for {child}')
        # A stable child-dependent order; no random tuning on outcomes.
        pool.sort(key=lambda p:sha256((child+'\0'+p).encode()).hexdigest())
        result[child]=pool[:count]
    return result


def mean(xs:list[float])->float|None:
    return float(np.mean(xs)) if xs else None


def fit_native(path:Path,seed:int,epochs:int,dim:int):
    import torch
    import mowl
    from mowl.datasets import PathDataset,Dataset
    from mowl.models import ELEmbeddings
    seed_everything(seed)
    # NOTE: init_jvm is invoked once in main; not here.
    dataset=Dataset(PathDataset(str(path.resolve())).ontology)
    model=ELEmbeddings(dataset,embed_dim=dim,margin=0.1,reg_norm=1.0,
                       learning_rate=0.001,batch_size=128,model_filepath=None,device='cpu')
    idx={str(iri):int(n) for iri,n in model.class_index_dict.items()}
    model.train(epochs=epochs,validate_every=epochs)
    embedded=model.module.class_embed.weight.detach().cpu().clone()
    if not torch.isfinite(embedded).all() or embedded.shape[0]!=len(idx):
        raise RuntimeError('mOWL returned invalid embedding matrix/index')
    return idx,embedded,{'mowl_version':getattr(mowl,'__version__','unknown'),
                         'torch_version':torch.__version__}


def run(root:Path,folds:list[dict],inputs:dict,snapshot:dict[str,str],
        native_epochs:int,dim:int,seed:int,projection_epochs:int,
        output_tag:str='rail_mowl_v8',reviewed_preparation:dict|None=None):
    import torch
    import mowl
    mowl.init_jvm('2g')
    cfg={**PROJECTION,'epochs':projection_epochs}
    output=root/'reports/experiments'/output_tag
    vectors=root/'generated/experiments'/output_tag
    vectors.mkdir(parents=True,exist_ok=True)
    total=[]
    for fold in folds:
        i=fold['fold']
        if digest(fold['training_path'])!=fold['train_sha256']:
            raise RuntimeError(f'Fold {i}: training file hash changed since preparation')
        if set(fold['heldout']) & (fold['training'] | closure(fold['training'])):
            raise RuntimeError(f'Fold {i}: training assertions entail a held-out target')
        log(f'Fold {i}/4: fitting ELEmbeddings ({native_epochs} epochs)')
        idx,native,versions=fit_native(fold['training_path'],seed+i,native_epochs,dim)
        required={iri for edge in fold['heldout'] for iri in edge}
        required|={iri for edge in fold['training'] for iri in edge}
        missing=required-set(idx)
        if missing:
            raise RuntimeError(f'Fold {i}: missing mOWL indexed classes: {sorted(missing)[:8]}')
        control_iris=controls(inputs['edges'],tuple(fold['heldout']))
        for iris in control_iris.values():
            if not set(iris).issubset(idx):
                raise RuntimeError(f'Fold {i}: missing controls from mOWL index')
        native_scores=score(native,fold['heldout'],idx)
        log(f'Fold {i}/4: fitting frozen-embedding residual MLP ({projection_epochs} epochs)')
        projector,loss=train_projection(native,idx,fold['training'],cfg,seed+100+i)
        with torch.no_grad():
            projected=projector(native)
        projected_scores=score(projected,fold['heldout'],idx)
        control_pairs=[(c,p) for c,_ in fold['heldout'] for p in control_iris[c]]
        native_controls=score(native,control_pairs,idx)
        projected_controls=score(projected,control_pairs,idx)
        rows=[]
        for (child,parent),ns,ps in zip(fold['heldout'],native_scores,projected_scores):
            this_control=[j for j,(c,p) in enumerate(control_pairs) if c==child]
            rows.append({'fold':i,'child':child,'expected_parent':parent,
                         **fold['support'][child],
                         'native_cosine':ns,'projected_cosine':ps,
                         'native_control_mean':mean([native_controls[j] for j in this_control]),
                         'projected_control_mean':mean([projected_controls[j] for j in this_control]),
                         'native_control_auc':rank_auc([ns],[native_controls[j] for j in this_control]),
                         'projected_control_auc':rank_auc([ps],[projected_controls[j] for j in this_control])})
        for j,(child,parent) in enumerate(control_pairs):
            rows.append({'fold':i,'child':child,'expected_parent':'',
                         'control_parent':parent,'native_control_cosine':native_controls[j],
                         'projected_control_cosine':projected_controls[j],
                         'record_type':'UNASSERTED_PARENT_CONTROL'})
        pos_native=mean(native_scores)
        pos_projected=mean(projected_scores)
        control_native=mean(native_controls)
        control_projected=mean(projected_controls)
        fold_metrics={'fold':i,'version':VERSION,'native_mean':pos_native,
                      'projected_mean':pos_projected,'native_control_mean':control_native,
                      'projected_control_mean':control_projected,
                      'native_positive_control_gap':pos_native-control_native,
                      'projected_positive_control_gap':pos_projected-control_projected,
                      'native_auc':rank_auc(native_scores,native_controls),
                      'projected_auc':rank_auc(projected_scores,projected_controls),
                      'native_epochs':native_epochs,'projection_config':cfg,
                      'training_losses':loss,'library_versions':versions,
                      'heldout_rows':[r for r in rows if 'expected_parent' in r and r['expected_parent']],
                      'approved_anchor_count':len(inputs.get('reviewed_anchors',set())),
                      'named_holdout_leakage_detected':False}
        save_json(output/f'fold_{i}_metrics.json',fold_metrics)
        torch.save({'state_dict':projector.state_dict(),'config':cfg,
                    'fold':i,'train_sha256':fold['train_sha256']},
                   vectors/f'fold_{i}_projection.pt')
        ordered=sorted(idx)
        with (vectors/f'fold_{i}_vectors.npz').open('wb') as f:
            np.savez_compressed(f,iris=np.asarray(ordered,dtype='U'),
                native=np.asarray([native[idx[iri]].numpy() for iri in ordered],dtype=np.float32),
                projected=np.asarray([projected[idx[iri]].numpy() for iri in ordered],dtype=np.float32))
        log(f"Fold {i}/4: native={pos_native:.4f}; projected={pos_projected:.4f}; "
            f"control gap={pos_projected-control_projected:.4f}")
        total.extend(rows)
    save_csv(output/'all_pair_scores.csv',total,
             ['fold','child','expected_parent','support_status','training_outgoing','training_incoming',
              'native_cosine','projected_cosine','native_control_mean',
              'projected_control_mean','native_control_auc','projected_control_auc',
              'control_parent','native_control_cosine','projected_control_cosine','record_type'])
    positive=[r for r in total if r.get('expected_parent')]
    controls_rows=[r for r in total if r.get('record_type')=='UNASSERTED_PARENT_CONTROL']
    npv=[r['native_cosine'] for r in positive]
    ppv=[r['projected_cosine'] for r in positive]
    nctrl=[r['native_control_cosine'] for r in controls_rows]
    pctrl=[r['projected_control_cosine'] for r in controls_rows]
    gap=mean(ppv)-mean(pctrl)
    auc=rank_auc(ppv,pctrl)
    counts=dict(Counter(r['support_status'] for r in positive))
    mean_gate=mean(ppv)>=TARGET
    control_gate=(gap>=0.15 and auc>=0.70)
    summary={'version':VERSION,
             'status':'TARGET_AND_CONTROLS_MET_DIAGNOSTIC_ONLY' if mean_gate and control_gate else 'TARGET_OR_CONTROLS_NOT_MET',
             'threshold':TARGET,'native_8_class_mean':mean(npv),
             'projected_8_class_mean':mean(ppv),
             'native_unasserted_control_mean':mean(nctrl),
             'projected_unasserted_control_mean':mean(pctrl),
             'native_positive_control_gap':mean(npv)-mean(nctrl),
             'projected_positive_control_gap':gap,
             'native_auc':rank_auc(npv,nctrl),'projected_auc':auc,
             'target_0_70_met':mean_gate,'control_gate_met':control_gate,
             'support_counts':counts,'holdout_count':len(positive),
             'negative_sampling_used':False,'validation_used_in_training_loss':False,
             'tuned_on_heldout':False,'original_pipeline_modified':False,
             'source_import_validation_reused_as_context_training':True,
             'unasserted_parent_controls_are_not_known_false_owl_axioms':True,
             'caveat':'Cold-start rail classes may have zero gradient support. High similarity is not OWL logical validity or semantic validation.',
             'inputs_sha256':snapshot,'folds_completed':4,
             'experiment_mode':'APPROVED_SEVEN_BROADER_ANCHORS' if reviewed_preparation else 'BASELINE',
             'approved_anchor_count':len(inputs.get('reviewed_anchors',set())),
             'review_file_sha256':reviewed_preparation.get('review_file_sha256') if reviewed_preparation else None,
             'named_subclass_leakage_detected':False,
             'full_owl_semantic_entailment_checked':False}
    if immutability_snapshot(root)!=snapshot:
        raise RuntimeError('READ-ONLY Project 6 upstream inputs changed during experiment')
    save_json(output/'eight_class_summary.json',summary)
    return summary


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--prepare-only',action='store_true',help='Check all folds and write only independent fold data; mOWL is not loaded')
    ap.add_argument('--reviewed-anchors',type=Path,default=None,help='JSON record of exactly the seven approved anchors; writes isolated reviewed outputs')
    ap.add_argument('--native-epochs',type=int,default=DEFAULT_NATIVE_EPOCHS)
    ap.add_argument('--projection-epochs',type=int,default=PROJECTION['epochs'])
    ap.add_argument('--seed',type=int,default=101)
    ap.add_argument('--embed-dim',type=int,default=EMBED_DIM)
    a=ap.parse_args()
    if a.native_epochs<1 or a.projection_epochs<1 or a.embed_dim<1:
        ap.error('Epoch counts and embedding dimensions must be positive')
    root=a.project_root.resolve()
    try:
        snapshot=immutability_snapshot(root)
        log('Verifying Project 6 Step 5A input hashes and eight audited rail assertions')
        inputs=load_split(root)
        if a.reviewed_anchors:
            folds,prep,inputs=prepare_approved(root,inputs,snapshot,a.reviewed_anchors)
            run_tag=REVIEWED_RUN_ID
            log('Seven human-approved broader superclass anchors integrated into four isolated folds')
        else:
            folds,prep=prepare(root,inputs,snapshot)
            run_tag='rail_mowl_v8'
        if immutability_snapshot(root)!=snapshot:
            raise RuntimeError('Project 6 upstream files changed during preparation')
        log('Four folds prepared; 2 held-out rail classes per fold; named reachability leakage check PASS')
        if a.prepare_only:
            log('PREPARED ONLY: no model trained; see reports/experiments/'+run_tag+'/preparation.json')
            return 0
        summary=run(root,folds,inputs,snapshot,a.native_epochs,a.embed_dim,a.seed,a.projection_epochs,
                    output_tag=run_tag,reviewed_preparation=prep if a.reviewed_anchors else None)
        log('8-class native mean: '+str(summary['native_8_class_mean']))
        log('8-class projected mean: '+str(summary['projected_8_class_mean']))
        log('Result: '+summary['status'])
        log('See reports/experiments/'+run_tag+'/eight_class_summary.json')
        return 0
    except Exception as exc:
        print('RAIL MOWL V8 EXPERIMENT FAILED: '+str(exc),file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
