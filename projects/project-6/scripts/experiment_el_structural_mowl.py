"""Project 6 Step 5D: audit retained OWL EL structure and train an isolated joint mOWL experiment.

Requires Project 6 Steps 1-5B (80/20 split) and the preceding
scripts/experiment_joint_rail_mowl.py (Project 5 transfer). Never changes source,
refined ontology, Step 5A/B artifacts, or the main run_all pipeline.

Modes:
    python -u scripts/experiment_el_structural_mowl.py --project-root . --diagnose-only
    python -u scripts/experiment_el_structural_mowl.py --project-root . --development-only --configs 1 --tune-epochs 50
    python -u scripts/experiment_el_structural_mowl.py --project-root . --configs 4 --tune-epochs 250 --final-epochs 700
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import random
import subprocess
import sys
import time

from rdflib import BNode, Graph, URIRef
from rdflib.collection import Collection
from rdflib.namespace import RDF, RDFS, OWL, XSD

import experiment_joint_rail_mowl as p5

VERSION='project6-el-structural-development-only-v3'
SUPPORTED={"named_to_some", "some_to_named", "intersection_to_named", "nested_el"}


def progress(stage:str):
    """Print immediately; silence is not mistaken for a hung diagnostic."""
    print(f"[EL structural {time.strftime('%H:%M:%S')}] {stage}", file=sys.stderr, flush=True)


def sig_key(sig) -> str:
    return json.dumps(sig, ensure_ascii=True, separators=(',', ':'), sort_keys=False)


def ast_from_graph(g: Graph, node, seen=frozenset(), depth=0):
    """Deterministic structural key for a *previously approved* EL expression.

    This is only called on expressions accepted by `expression` and is not a
    fallback OWL parser. Returns a canonical tree independent of BNode IDs and
    RDF list order (which has no semantic significance for intersectionOf).
    """
    if isinstance(node, URIRef):
        return ('named', str(node))
    if not isinstance(node, BNode) or node in seen or depth>18:
        raise ValueError('Unexpected cyclic or invalid EL expression during serialization')
    seen = seen | {node}
    intersection = g.value(node, OWL.intersectionOf)
    if intersection is not None:
        items = list(Collection(g,intersection))
        if len(items)<2 or len(items)>30:
            raise ValueError('Invalid intersection expression during serialization')
        children = tuple(sorted((ast_from_graph(g,item,seen,depth+1) for item in items),key=sig_key))
        return ('and', children)
    prop = g.value(node,OWL.onProperty)
    filler = g.value(node,OWL.someValuesFrom)
    if not isinstance(prop,URIRef) or filler is None:
        raise ValueError('Unsupported EL expression reached serializer')
    return ('some',str(prop),ast_from_graph(g,filler,seen,depth+1))


def deterministic_bnode(kind:str,signature) -> BNode:
    digest=hashlib.sha256((kind + '\0' + sig_key(signature)).encode('utf-8')).hexdigest()
    return BNode(kind + digest)


def emit_el_expression(g:Graph,signature):
    """Build the exact EL class expression using deterministic structure-based IDs.

    Unlike whole-graph RDF canonicalization, this is linear in the expression
    size; identical expressions reuse the same blank node, preserving OWL meaning.
    """
    kind=signature[0]
    if kind=='named': return URIRef(signature[1])
    node=deterministic_bnode('ex',signature)
    if kind=='some':
        filler=emit_el_expression(g,signature[2])
        g.add((node,RDF.type,OWL.Restriction))
        g.add((node,OWL.onProperty,URIRef(signature[1])))
        g.add((node,OWL.someValuesFrom,filler))
    elif kind=='and':
        children=signature[1]
        g.add((node,RDF.type,OWL.Class))
        head=deterministic_bnode('li',(signature,0))
        g.add((node,OWL.intersectionOf,head))
        for i,child in enumerate(children):
            cur=deterministic_bnode('li',(signature,i))
            nxt=deterministic_bnode('li',(signature,i+1)) if i<len(children)-1 else RDF.nil
            g.add((cur,RDF.first,emit_el_expression(g,child)))
            g.add((cur,RDF.rest,nxt))
    else:
        raise ValueError(f'Unknown EL expression: {kind}')
    return node


def entry_signature(entry):
    if 'signature' in entry:
        return entry['signature']
    # Backward compatibility with entries constructed by older unit tests.
    g=Graph()
    for t in entry['triples']:g.add(t)
    lhs,rhs=next((s,o) for s,p,o in g if p==RDFS.subClassOf)
    return (ast_from_graph(g,lhs),ast_from_graph(g,rhs))


def expression(g: Graph, node, seen=frozenset(), depth=0):
    """Parse only sound named, intersection or object-some EL expressions.

    Returns (kind, resources, closure_triples) or None. Do not simplify disallowed
    hasValue/cardinality/data restrictions into a different axiom.
    """
    if isinstance(node, URIRef):
        if node not in (OWL.Thing, OWL.Nothing, RDFS.Literal) and not str(node).startswith(str(XSD)) and (node,RDF.type,RDFS.Datatype) not in g:
            return ('named', {node}, set())
        return None
    if not isinstance(node,BNode) or node in seen or depth>18:
        return None
    seen=seen|{node}
    props=set(g.predicates(node,None))
    # Skip anything not a legal structural EL expression; rdf:type Class/Restriction OK.
    allowed={RDF.type,OWL.onProperty,OWL.someValuesFrom,OWL.intersectionOf,
             RDFS.subClassOf,OWL.equivalentClass}
    if props-allowed:
        return None
    prop=g.value(node,OWL.onProperty)
    val=g.value(node,OWL.someValuesFrom)
    intersection=g.value(node,OWL.intersectionOf)
    if intersection is not None:
        if prop is not None or val is not None:
            return None
        try: members=list(Collection(g,intersection))
        except Exception: return None
        if len(members)<2 or len(members)>30:
            return None
        triples={(node,OWL.intersectionOf,intersection)}
        items=set()
        try:
            coll=Collection(g,intersection)
            head=intersection
            for _ in members:
                first=g.value(head,RDF.first); rest=g.value(head,RDF.rest)
                if first is None or rest is None:return None
                triples|={(head,RDF.first,first),(head,RDF.rest,rest)}
                head=rest
            if head!=RDF.nil:return None
        except Exception: return None
        for m in members:
            e=expression(g,m,seen,depth+1)
            if e is None:return None
            items.update(e[1]);triples.update(e[2])
        triples.add((node,RDF.type,OWL.Class))
        return ('intersection',items,triples)
    if isinstance(prop,URIRef) and val is not None:
        if (prop,RDF.type,OWL.DatatypeProperty) in g or (prop,RDF.type,OWL.AnnotationProperty) in g:
            return None
        if g.value(node,OWL.intersectionOf) is not None:return None
        filler=expression(g,val,seen,depth+1)
        if filler is None:return None
        triples={(node,RDF.type,OWL.Restriction),(node,OWL.onProperty,prop),(node,OWL.someValuesFrom,val)}|filler[2]
        return ('some',set(filler[1]),triples)
    return None


def extract_el_axioms(g: Graph):
    """Collect original anonymous-expression GCIs, never fabricated subclass triples.

    Reject ambiguous OWL structures; no axioms are constructed from annotation
    similarity. Duplicate triples are deduped only after extracting axioms.
    """
    entries=[];reasons=Counter()
    def add(s,o,relation):
        # Named subclass pairs already belong to the 80/20 split; do not re-add.
        if isinstance(s,URIRef) and isinstance(o,URIRef):
            reasons['named_pairs_left_to_split']+=1;return
        e1=expression(g,s);e2=expression(g,o)
        if not e1 or not e2:
            reasons['unsupported_or_non_el_expression']+=1;return
        if e1[0]=='named' and e2[0]=='some':kind='named_to_some'
        elif e1[0]=='some' and e2[0]=='named':kind='some_to_named'
        elif e1[0]=='intersection' and e2[0]=='named':kind='intersection_to_named'
        else:kind='nested_el'
        triples={(s,RDFS.subClassOf,o)}|e1[2]|e2[2]
        entries.append({'kind':kind,'classes':set(e1[1])|set(e2[1]),
                        'left_classes':set(e1[1]),'triples':triples,
                        'representation':relation,
                        'signature':(ast_from_graph(g,s),ast_from_graph(g,o))})
    for s,o in g.subject_objects(RDFS.subClassOf):add(s,o,'subClassOf')
    for s,o in g.subject_objects(OWL.equivalentClass):
        add(s,o,'equivalent_forward');add(o,s,'equivalent_reverse')
    return entries,reasons


def collect_approved_owl(root:Path, inputs:dict):
    """Load original source, imports and refined; verify import SHA before reading.

    The imported source may have non-EL axioms: those are audited, not blindly fed
    into ELEmbeddings. Provenance is tied to Step 5A's manifest.
    """
    files=[('source',inputs['source']),('refined',inputs['refined'])]
    for relative,sha in sorted(inputs['split'].get('import_hashes',{}).items()):
        path=root/relative
        if not path.is_file() or p5.digest(path)!=sha:
            raise ValueError(f'Local import provenance invalid: {relative}')
        files.append((relative,path))
    all_entries=[];summary={};reason_totals=Counter()
    for i,(kind,path) in enumerate(files,1):
        progress(f'Parsing {i}/{len(files)}: {kind} ({path.stat().st_size:,} bytes)')
        g=p5.rdf(path)
        entries,reasons=extract_el_axioms(g)
        summary[kind]={
            'triples':len(g),'anonymous_or_expression_axioms_considered':sum(1 for x in g.subject_objects(RDFS.subClassOf) if isinstance(x[0],BNode) or isinstance(x[1],BNode)) + sum(2 for _ in g.subject_objects(OWL.equivalentClass)),
            'retained_syntactic_el_axioms':len(entries),
            'kinds':dict(sorted(Counter(e['kind'] for e in entries).items())),
            'skips':dict(sorted(reasons.items()))}
        for e in entries:
            e['origin']=kind
            all_entries.append(e)
        reason_totals.update(reasons)
        progress(f'Parsed {kind}: {len(g):,} triples, {len(entries)} syntactic EL axioms')
    return all_entries,{'file_audits':summary,'reasons':dict(sorted(reason_totals.items()))}


def select_axioms(entries,heldouts:set[tuple[str,str]],*,suppress_heldout_children=True):
    """Conservatively exclude any complex axiom mentioning validation children.

    This reduces obvious symbolic leakage, but is not a complete OWL entailment
    proof. The report explicitly labels the limitation; a reasoner should check
    it before any claim of independently verified OWL EL holdouts.
    """
    held_children={URIRef(str(a)) for a,_ in heldouts}
    selected=[];excluded=Counter()
    for entry in entries:
        if suppress_heldout_children and entry['classes'] & held_children:
            excluded['mentions_validation_child']+=1;continue
        selected.append(entry)
    return selected,excluded


def build_training_graph(base:Graph,training_pairs:set[tuple[str,str]],selected:list,destination:Path):
    """Deterministic EL projection without expensive whole-graph RDF canonicalization.

    All blank-node identifiers are derived from the structural expression tree.
    Exact round-trip is checked using parsed EL signature sets and named pairs,
    *not* the potentially exponential `isomorphic()` algorithm.
    """
    progress(f'Writing {destination.name}: {len(training_pairs)} named pairs and {len(selected)} supported EL axioms')
    g=Graph()
    for s,p,o in base:
        if p==RDF.type and o==OWL.Class:g.add((s,p,o))
    for s,o in sorted(training_pairs):g.add((URIRef(s),RDFS.subClassOf,URIRef(o)))
    signatures=set()
    for entry in selected:
        signature=entry_signature(entry)
        signatures.add(signature)
        left=emit_el_expression(g,signature[0])
        right=emit_el_expression(g,signature[1])
        g.add((left,RDFS.subClassOf,right))
    props={o for s,p,o in g if p==OWL.onProperty and isinstance(o,URIRef)}
    for prop in sorted(props):g.add((prop,RDF.type,OWL.ObjectProperty))
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix('.ttl.tmp')
    lines=sorted(f'{s.n3()} {p.n3()} {o.n3()} .' for s,p,o in g)
    tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tmp.replace(destination)
    progress(f'Validating {destination.name}: {len(g)} triples (no graph-isomorphism search)')
    parsed=p5.rdf(destination)
    actual,why=extract_el_axioms(parsed)
    found={entry_signature(e) for e in actual}
    if len(parsed)!=len(g) or found!=signatures:
        raise AssertionError('EL structural roundtrip failed: triple count or expression signatures differ')
    if p5.named_pairs(parsed)!=training_pairs:
        raise AssertionError('A held-out or unexpected named pair appeared in structural training')
    p5.write_json(destination.with_suffix('.meta.json'),{
        'version':VERSION,'extra_owl_axioms':len(selected),'triple_count':len(g),
        'unique_el_structural_axioms':len(signatures)})
    progress(f'Verified {destination.name}')
    return {'triple_count':len(g),'additional_owl_axioms':len(selected),
            'named_training_pairs':len(training_pairs),
            'restriction_count':len(list(g.subjects(OWL.onProperty,None))),
            'sha256':p5.digest(destination)}


def diagnose(root:Path,seed:int,dev_fraction:float):
    """Fast **read-only** EL diagnostic; does not build or canonicalize datasets.

    This deliberately does not invoke prepare() or OWLAPI/mOWL, so --diagnose-only
    is usable when training graph construction needs further diagnosis.
    """
    started=time.monotonic()
    progress('1/5 Validate original 80/20 split and SHA-256 provenance')
    inputs=p5.check_inputs(root)
    base_checks(root,inputs)
    progress('2/5 Create internal development split (no graph serialization)')
    dev_train,dev_hold=p5.internal_dev_split(inputs['train'],seed,dev_fraction)
    progress('3/5 Inspect EL expressions in original source, refined and imports')
    entries,audit=collect_approved_owl(root,inputs)
    progress('4/5 Apply outer and development holdout filters')
    outer,outer_drop=select_axioms(entries,inputs['valid'])
    inner,inner_drop=select_axioms(entries,inputs['valid']|dev_hold)
    base_checks(root,inputs)
    report={'version':VERSION,'status':'PASS_EL_DIAGNOSE_ONLY',
        'elapsed_seconds':round(time.monotonic()-started,3),
        'diagnose_only':True, 'training_graphs_written':False,
        'normal_forms_counted':False,
        'input_sha256':{k:inputs['split'][k] for k in ('source_sha256','refined_sha256','train_sha256','valid_sha256')},
        'split':{'outer_training':len(inputs['train']),'outer_holdout':len(inputs['valid']),
            'outer_rail_holdout':len(inputs['rail_hold']),
            'development_training':len(dev_train),'development_holdout':len(dev_hold)},
        'source_audit':audit,
        'outer':{'additional_owl_axioms':len(outer),
                 'excluded':dict(outer_drop),'source_el_kinds':dict(Counter(e['kind'] for e in outer))},
        'development':{'additional_owl_axioms':len(inner),
                 'excluded':dict(inner_drop),'source_el_kinds':dict(Counter(e['kind'] for e in inner))},
        'limitations':['Syntax-only EL diagnostic. Does not certify OWL logical nonleakage.',
                       'Actual mOWL normal-form counts require the subsequent training run.']}
    p5.write_json(root/'reports/step5_el_structure_diagnostic.json',report)
    progress('5/5 Diagnostic complete: reports/step5_el_structure_diagnostic.json')
    return report


def real_gci_counts(path:Path):
    """Ask mOWL's real ELNormalizer to report GCI types; never infer from RDF alone."""
    from mowl.datasets import Dataset,PathDataset
    from mowl.datasets.el import ELDataset
    ds=Dataset(PathDataset(str(path.resolve())).ontology)
    mapping=ELDataset(ds.ontology).get_gci_datasets()
    return {str(k):len(v) for k,v in mapping.items()}


def base_checks(root:Path, inputs):
    checks=[(inputs['source'],'source_sha256'),(inputs['refined'],'refined_sha256'),
            (inputs['path_train'],'train_sha256'),(inputs['path_valid'],'valid_sha256')]
    for path,key in checks:
        if p5.digest(path)!=inputs['split'][key]:
            raise RuntimeError(f'Immutable Project 6 input was changed: {path}')
    for rel,sha in inputs['split'].get('import_hashes',{}).items():
        if p5.digest(root/rel)!=sha:raise RuntimeError(f'Import changed: {rel}')


def prepare(root:Path,seed:int,dev_fraction:float,*,development_only:bool=False):
    progress(('Development-only' if development_only else 'Full') + ' preparation: checking inputs')
    inputs=p5.check_inputs(root)
    dev_train,dev_hold=p5.internal_dev_split(inputs['train'],seed,dev_fraction)
    progress('Preparation: loading original training taxonomy')
    base=p5.rdf(inputs['path_train'])
    entries,source_audit=collect_approved_owl(root,inputs)
    progress('Preparation: filtering EL structures against validation children')
    outer_select,outer_drop=select_axioms(entries,inputs['valid'])
    inner_select,inner_drop=select_axioms(entries,inputs['valid']|dev_hold)
    dest_outer=root/'generated/mowl_el_structural_outer_train.ttl'
    dest_dev=root/'generated/mowl_el_structural_dev_train.ttl'
    # The development-only mode must not even *build* the outer training graph.
    # In particular, it must not overwrite an existing outer graph or vector file.
    if development_only:
        outer_info={'additional_owl_axioms':len(outer_select),
                    'training_graph_written':False}
    else:
        outer_info=build_training_graph(base,inputs['train'],outer_select,dest_outer)
        outer_info['training_graph_written']=True
    inner_info=build_training_graph(base,dev_train,inner_select,dest_dev)
    inner_info['training_graph_written']=True
    # Only structural named-class reachability can be checked without OWL reasoning.
    # With anonymous class GCIs, a full EL reasoner is needed to rule out inferences.
    if not development_only and inputs['valid']&p5.closure(p5.named_pairs(p5.rdf(dest_outer))):
        raise ValueError('Outer holdout transitively entailed by named training taxonomy')
    if dev_hold&p5.closure(p5.named_pairs(p5.rdf(dest_dev))):
        raise ValueError('Inner holdout transitively entailed by named training taxonomy')
    report={'version':VERSION,'status':('EL_STRUCTURAL_DEVELOPMENT_PROJECTION_PREPARED' if development_only else 'EL_STRUCTURAL_PROJECTION_PREPARED'),
            'input_sha256':{k:inputs['split'][k] for k in ('source_sha256','refined_sha256','train_sha256','valid_sha256')},
            'split':{'outer_training':len(inputs['train']),'outer_holdout':len(inputs['valid']),
                     'outer_rail_holdout':len(inputs['rail_hold']),
                     'development_training':len(dev_train),'development_holdout':len(dev_hold)},
            'source_audit':source_audit,
            'outer':{**outer_info,'excluded':dict(outer_drop),'source_el_kinds':dict(Counter(e['kind'] for e in outer_select)),
                     **({} if development_only else {'path':str(dest_outer.relative_to(root))})},
            'development':{**inner_info,'excluded':dict(inner_drop),'source_el_kinds':dict(Counter(e['kind'] for e in inner_select)),
                           'path':str(dest_dev.relative_to(root))},
            'leakage_note':'No directly or transitively present named holdouts; exclusion of anonymous axioms referencing heldout children is conservative, NOT a full OWL entailment check.',
            'limitations':['Some supported OWL EL expressions are excluded to reduce holdout leakage.',
                           'Nested class expressions may be normalized by JCEL into multiple auxiliary GCIs.',
                           'Unsupported hasValue/cardinality/datatype axioms are omitted without approximation.',
                           'No full semantic leakage guarantee without separate OWL reasoner.']}
    p5.write_json(root/'reports/step5_el_structure_diagnostic.json',report)
    base_checks(root,inputs)
    return inputs,dev_train,dev_hold,report


def run(root:Path,seed=42,dev_fraction=.16,configs=4,tune_epochs=250,final_epochs=700,
        max_inferred=600,controls=120,text_width=256,threshold=.70,development_only=False):
    if not (0<threshold<=1) or min(configs,tune_epochs,controls,max_inferred,text_width)<1 or (not development_only and final_epochs<1):
        raise ValueError('Invalid hyperparameters')
    inputs,dev_train,dev_hold,diag=prepare(root,seed,dev_fraction,development_only=development_only)
    split_to_check='development' if development_only else 'outer'
    if diag[split_to_check]['additional_owl_axioms']==0:
        raise ValueError(f'No supported non-GCI0 EL axioms survived {split_to_check} leakage safeguards; inspect diagnostic instead of pretending richer training exists')
    import mowl
    mowl.init_jvm('2g')
    dest_dev=root/'generated/mowl_el_structural_dev_train.ttl'
    dest_outer=root/'generated/mowl_el_structural_outer_train.ttl'
    normalized={'development':real_gci_counts(dest_dev)}
    if not development_only:
        normalized['outer']=real_gci_counts(dest_outer)
    diag['normalized_gci_counts']=normalized
    p5.write_json(root/'reports/step5_el_structure_diagnostic.json',diag)
    if sum(normalized['development'].get(k,0) for k in ('gci1','gci2','gci3'))==0:
        raise RuntimeError('mOWL normalized no GCI1/GCI2/GCI3 in development training graph; STOP.')
    if not development_only and sum(normalized['outer'].get(k,0) for k in ('gci1','gci2','gci3'))==0:
        raise RuntimeError('mOWL normalized no GCI1/GCI2/GCI3 in outer graph; STOP.')
    texts=p5.collect_text(root,inputs)
    configurations=[dict(dimension=dim,lambda_cos=lam,semantic_scale=scale,
                        learning_rate=.001,dispersion_weight=disp)
                    for dim,lam,scale,disp in [(50,.25,.4,.02),(32,.25,.4,.02),
                                               (50,.75,.4,.02),(50,.25,.15,.02),
                                               (50,.25,.4,0.),(32,.75,.4,.02),
                                               (50,1.5,.4,.02),(32,.25,.15,.02)]]
    # fixed configs: no evaluation-dependent test-set tuning.
    configs=min(configs,len(configurations))
    train_forbidden=inputs['valid']|dev_hold|inputs['train']
    sweep=[];chosen=None
    for i,cfg in enumerate(configurations[:configs],1):
        print(f'[EL structural] development model {i}/{configs} {cfg}',flush=True)
        out,_,_,_,_=p5.train_once(root,dest_dev,dev_train,dev_hold,train_forbidden,texts,
                                   cfg,tune_epochs,seed,max_inferred,controls,text_width,f'el_dev_{i}')
        out['selection_score']=p5.select_score(out['mean_cosine'],out['positive_control_gap'])
        sweep.append(out)
        if chosen is None or out['selection_score']>chosen['selection_score']:chosen=out
    cfg=chosen['config']
    if development_only:
        # No outer fit, outer holdout scoring, outer vector save, or comparison
        # report. All model selection and diagnostics use inner development only.
        base_checks(root,inputs)
        report={'version':VERSION,'status':'DEVELOPMENT_ONLY_COMPLETE',
                'mode':'development_only','seed':seed,'tune_epochs':tune_epochs,
                'threshold_reference_only':threshold,
                'outer_holdout_evaluated':False,'outer_final_fit_run':False,
                'outer_training_graph_written':False,'outer_vectors_written':False,
                'development_holdout_pairs':len(dev_hold),
                'development_training_pairs':len(dev_train),
                'normal_forms':normalized,
                'chosen_development_config':cfg,'chosen_dev_result':chosen,
                'development_sweep':sweep,
                'input_sha256':diag['input_sha256'],
                'development_training_graph_sha256':diag['development']['sha256'],
                'report_scope':'Internal development only; outer holdout is not evaluated.',
                'warnings':['No final holdout or rail performance is reported by this run.',
                            'Do not interpret development cosine as proof of the 0.70 outer requirement.',
                            'Source text features may implicitly encode taxonomy information.',
                            'An OWL reasoner is needed to certify semantic nonleakage.']}
        path=root/'reports/step5_el_structural_development.json'
        p5.write_json(path,report)
        progress('Development-only complete: reports/step5_el_structural_development.json (no outer fit)')
        return report
    print(f'[EL structural] outer final fit, config selected on internal development ONLY: {cfg}',flush=True)
    output,rows,control_rows,mat,index=p5.train_once(
        root,dest_outer,inputs['train'],inputs['valid'],inputs['valid']|inputs['train'],texts,
        cfg,final_epochs,seed,max_inferred,controls,text_width,'el_outer_final')
    rail=[r for r in rows if (r['child'],r['parent']) in inputs['rail_hold']]
    sources=[r for r in rows if (r['child'],r['parent']) in inputs['base_hold']]
    stats_rail=p5.stats(rail);stats_all=p5.stats(rows);stats_source=p5.stats(sources)
    blocked=inputs['train']|inputs['valid']|p5.closure(inputs['train'])
    matched=p5.matched_child_controls(index,inputs['rail_hold'],blocked,3,seed+101)
    matched_rows=p5.similarity_rows(mat,matched,index,'el_outer_rail_controls')
    controls_stats=p5.stats(control_rows);matched_stats=p5.stats(matched_rows)
    auc=p5.auc_pos_control(rows,control_rows);auc_rail=p5.auc_pos_control(rail,matched_rows)
    gap=round(stats_all['mean']-controls_stats['mean'],6) if controls_stats['mean'] is not None else None
    gap_rail=round(stats_rail['mean']-matched_stats['mean'],6) if matched_stats['mean'] is not None else None
    pass_threshold=stats_rail['mean']>=threshold and stats_all['mean']>=threshold
    pass_controls=gap is not None and gap>=.15 and auc is not None and auc>=.70 and gap_rail is not None and gap_rail>=.15 and auc_rail is not None and auc_rail>=.70
    # Require actual normalization and good controls. But no final assertion of semantic
    # nonleakage until a separate OWL consequence test is performed.
    status='PASS_METRIC_EXPERIMENT_REQUIRES_OWL_LEAKAGE_REVIEW' if pass_threshold and pass_controls else 'FAIL_HOLDOUT_COSINE_OR_CONTROLS'
    base_checks(root,inputs)
    import numpy as np
    target=root/'generated/mowl_el_structural_vectors.npz'
    ordered=sorted(index)
    temp=target.with_name(target.name+'.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,iris=np.asarray(ordered,dtype='U'),vectors=np.asarray([mat[index[iri]].cpu().numpy() for iri in ordered],dtype=np.float32))
    temp.replace(target)
    report={'version':VERSION,'status':status,'threshold':threshold,'metric_threshold_met':pass_threshold,
           'control_guard_met':pass_controls,'rail_holdout':stats_rail,'source_import_holdout':stats_source,
           'all_holdout':stats_all,'all_heldout_control':controls_stats,'all_auc':auc,'all_positive_control_gap':gap,
           'rail_matched_controls':matched_stats,'rail_matched_auc':auc_rail,'rail_positive_control_gap':gap_rail,
           'individual_rail_scores':rail,'normal_forms':normalized,
           'chosen_development_config':cfg,'chosen_dev_result':chosen,'development_sweep':sweep,
           'final_fit':output,'baseline_mowl':{'rail_mean':inputs['baseline']['mean_heldout_rail_cosine'],
                                             'source_import_mean':inputs['baseline']['mean_heldout_source_import_cosine']},
           'input_sha256':diag['input_sha256'],
           'training_dataset_hashes':{'outer':diag['outer']['sha256'],'development':diag['development']['sha256']},
           'vector_file':str(target.relative_to(root)),'vector_sha256':p5.digest(target),
           'leakage_assessment':'Named direct/transitive holdout absent; full OWL entailment not yet independently verified.',
           'outer_holdout_used_to_tune':False,
           'warnings':['This is a separate experiment, not proof Project 6 run_all meets cosine threshold.',
                       'Two rail holdouts are too small for stable out-of-sample claims.',
                       'Non-EL axioms (e.g. hasValue) are omitted, not approximated.',
                       'Unasserted controls are not logically proven false under open-world semantics.']}
    p5.write_json(root/'reports/step5_el_structural_comparison.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--dev-fraction',type=float,default=.16)
    parser.add_argument('--diagnose-only',action='store_true')
    parser.add_argument('--development-only',action='store_true',
                        help='Train/evaluate only internal development; no outer fit, holdout scores or vectors')
    parser.add_argument('--diagnose-timeout-seconds',type=int,default=120,
                        help='Stop --diagnose-only clearly if parsing takes too long (default 120)')
    parser.add_argument('--_diagnose-child',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--configs',type=int,default=4)
    parser.add_argument('--tune-epochs',type=int,default=250)
    parser.add_argument('--final-epochs',type=int,default=700)
    parser.add_argument('--max-inferred',type=int,default=600)
    parser.add_argument('--controls',type=int,default=120)
    parser.add_argument('--text-width',type=int,default=256)
    parser.add_argument('--threshold',type=float,default=.70)
    parser.add_argument('--strict',action='store_true')
    a=parser.parse_args()
    if a.diagnose_only and a.development_only:
        parser.error('--diagnose-only and --development-only are mutually exclusive')
    if a.development_only and a.strict:
        parser.error('--strict requires outer evaluation; omit it with --development-only')
    if a.diagnose_only and not a._diagnose_child:
        if a.diagnose_timeout_seconds<1:
            parser.error('--diagnose-timeout-seconds must be >= 1')
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--project-root',str(a.project_root.resolve()),
             '--seed',str(a.seed),'--dev-fraction',str(a.dev_fraction),'--diagnose-only','--_diagnose-child']
        progress(f'Starting bounded diagnostic (timeout={a.diagnose_timeout_seconds}s)')
        proc=subprocess.Popen(cmd)
        try:
            return proc.wait(timeout=a.diagnose_timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print('EL DIAGNOSTIC TIMEOUT: stopped after '+str(a.diagnose_timeout_seconds)+
                  ' seconds. Last [EL structural] progress line identifies the slow stage.',file=sys.stderr,flush=True)
            return 124
    try:
        if a.diagnose_only:
            d=diagnose(a.project_root.resolve(),a.seed,a.dev_fraction)
            print(json.dumps({'status':d['status'],'outer':d['outer'],'development':d['development'],
                              'diagnostic':'reports/step5_el_structure_diagnostic.json'},indent=2))
            return 0
        r=run(a.project_root.resolve(),seed=a.seed,dev_fraction=a.dev_fraction,configs=a.configs,
              tune_epochs=a.tune_epochs,final_epochs=a.final_epochs,max_inferred=a.max_inferred,
              controls=a.controls,text_width=a.text_width,threshold=a.threshold,
              development_only=a.development_only)
    except Exception as e:
        print(f'EL STRUCTURAL EXPERIMENT ERROR: {e}',file=sys.stderr)
        return 2
    if a.development_only:
        print(json.dumps({'status':r['status'],'development':r['chosen_dev_result'],
                          'normal_forms':r['normal_forms'],
                          'report':'reports/step5_el_structural_development.json'},indent=2))
        return 0
    print(json.dumps({'status':r['status'],'rail':r['rail_holdout'],'all':r['all_holdout'],
                      'normal_forms':r['normal_forms'],'report':'reports/step5_el_structural_comparison.json'},indent=2))
    return 1 if a.strict and r['status']!='PASS_METRIC_EXPERIMENT_REQUIRES_OWL_LEAKAGE_REVIEW' else 0


if __name__=='__main__':raise SystemExit(main())
