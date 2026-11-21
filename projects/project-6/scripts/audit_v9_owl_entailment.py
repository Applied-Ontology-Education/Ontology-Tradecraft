#!/usr/bin/env python3
"""Project 6: isolated HermiT OWL entailment audit of frozen mOWL v9.

TWO distinct logical contexts, tested separately on the same eight holdouts:
  named_v9: exactly the v9 outer-training RDF graph (named subclass only).
  full_context: source ontology + six local imports + refined rail TBox +
    approved anchors, excluding the two outer holdouts in the relevant fold.

Do NOT confuse full_context with the axiom set trained by mOWL v9.
No inputs are modified. No network imports are resolved; all local source/import
files are explicitly loaded, import mappings are checked, and owl:imports is
removed ONLY from the isolated materialized audit copy.

From projects/project-6:
    python -u scripts/audit_v9_owl_entailment.py --prepare-only
    python -u scripts/audit_v9_owl_entailment.py

Requires rdflib, mowl, jpype in .venv-mowl and a compatible HermiT JVM class.
If a complete reasoner is unavailable, FAIL instead of falsely reporting PASS.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

VERSION = 'project6-v9-hermit-dual-context-entailment-audit-v1'
BASE = 'https://example.org/ontologies/intel-ontology#'
BFO = 'http://purl.obolibrary.org/obo/'
AGENT = 'https://www.commoncoreontologies.org/AgentOntology/'
FOLDS = (
    ((BASE+'rail_passenger_role', AGENT+'passenger'),(BASE+'rail_reservation_information', AGENT+'reservation')),
    ((BASE+'railway_station', BFO+'BFO_0000040'),(BASE+'train', BFO+'BFO_0000040')),
    ((BASE+'rail_journey_plan', BFO+'IAO_0000030'),(BASE+'rail_ticket', AGENT+'ticket')),
    ((BASE+'passenger_train', BASE+'train'),(BASE+'rail_journey', BFO+'BFO_0000015')),
)
ANCHORS = frozenset({
    (BASE+'rail_reservation_information',BFO+'BFO_0000031'),
    (BASE+'railway_station',BFO+'BFO_0000004'),
    (BASE+'train',BFO+'BFO_0000004'),
    (BASE+'rail_journey_plan',BFO+'BFO_0000031'),
    (BASE+'rail_ticket',BFO+'IAO_0000178'),
    (BASE+'passenger_train',BFO+'BFO_0000040'),
    (BASE+'rail_journey',BFO+'BFO_0000003'),
})
IMPORTS = (
    'imports/AgentOntology.ttl', 'imports/EventOntology.ttl',
    'imports/InformationEntityOntology.ttl', 'imports/bfo-core.ttl',
    'imports/core.ttl','imports/iao-edit.owl',
)
OUT='reports/experiments/rail_mowl_v9_owl_audit'
GEN='generated/experiments/rail_mowl_v9_owl_audit'
V9PREP='reports/experiments/rail_mowl_v9_reviewed/preparation.json'
V9SUMMARY='reports/experiments/rail_mowl_v9_reviewed/eight_class_summary.json'
REVIEW='reports/experiments/rail_anchor_review/approved_anchors.json'


def log(s): print('[v9 OWL audit] '+str(s),flush=True)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def load(path):
    if not path.is_file(): raise FileNotFoundError(str(path))
    g=Graph()
    # RDFLib guesses the format from extension; .owl can be RDF/XML or Turtle.
    # Only parse formats actually supported by these shipped local ontology files.
    if path.suffix.lower()=='.owl':
        try: g.parse(str(path),format='xml')
        except Exception as ex:
            g=Graph()
            try: g.parse(str(path),format='turtle')
            except Exception as ex2: raise RuntimeError(f'Cannot parse {path} as RDF/XML or Turtle: {ex} / {ex2}') from ex2
    else: g.parse(str(path),format='turtle')
    return g

def named(g):
    return {(str(c),str(p)) for c,p in g.subject_objects(RDFS.subClassOf)
            if isinstance(c,URIRef) and isinstance(p,URIRef) and c!=p}

def closure(edges):
    parents=defaultdict(set)
    for c,p in edges: parents[c].add(p)
    seen_pairs=set()
    for c in {n for pair in edges for n in pair}:
        todo=deque(parents.get(c,()))
        seen={c}
        while todo:
            p=todo.popleft()
            if p in seen:continue
            seen.add(p)
            seen_pairs.add((c,p))
            todo.extend(parents.get(p,set())-seen)
    return seen_pairs

def write_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    tmp.replace(path)

def check_provenance(root):
    prep=json.loads((root/V9PREP).read_text(encoding='utf-8'))
    summary=json.loads((root/V9SUMMARY).read_text(encoding='utf-8'))
    review=json.loads((root/REVIEW).read_text(encoding='utf-8'))
    if summary.get('experiment_mode')!='NESTED_DEV_SEVEN_APPROVED_ANCHORS' or summary.get('configs_per_fold')!=10:
        raise ValueError('Not the final v9 10-configuration experiment')
    if summary.get('folds_completed')!=4 or summary.get('outer_test_sample_size')!=8:
        raise ValueError('Not the final 4-fold/8-holdout experiment')
    if summary.get('projection_fallback_to_native_folds')!=4:
        raise ValueError('Unexpected v9 selection state: recheck frozen baseline')
    if abs(summary.get('native_8_class_mean',-100)-0.6037136092782021)>1e-7 or abs(summary.get('native_auc',-1)-.953125)>1e-7:
        raise ValueError('v9 summary differs from frozen full-training baseline')
    if prep.get('mode')!='SEVEN_APPROVED_ANCHORS_NESTED_DEVELOPMENT' or len(prep.get('folds',[]))!=4:
        raise ValueError('Invalid v9 preparation report')
    if prep.get('review_file_sha256')!=digest(root/REVIEW) or summary.get('review_file_sha256')!=digest(root/REVIEW):
        raise RuntimeError('Approved superclass record changed')
    if review.get('human_review_status')!='SEVEN_APPROVED_FOR_EXPERIMENT_2026_11_19' or not review.get('target_holdouts_must_remain_excluded'):
        raise ValueError('Unexpected anchor review authorization')
    rows=review.get('approved_anchors',[])
    if len(rows)!=7 or {(r.get('child'),r.get('parent')) for r in rows}!=ANCHORS:
        raise ValueError('Approved anchor list does not match exactly seven reviewed axioms')
    for file, sha in summary['input_sha256'].items():
        if not (root/file).is_file() or digest(root/file)!=sha or prep.get('inputs_sha256',{}).get(file)!=sha:
            raise RuntimeError('Frozen source hash mismatch: '+file)
    for i,fold in enumerate(prep['folds'],1):
        if fold['fold']!=i or {(x['child'],x['parent']) for x in fold['heldout']}!=set(FOLDS[i-1]):
            raise ValueError(f'Changed holdout allocation in fold {i}')
        for tag in ('outer','inner'):
            file=root/fold[f'{tag}_training_path']
            if not file.is_file() or digest(file)!=fold[f'{tag}_sha256']:
                raise RuntimeError(f'v9 fold {i} {tag} training graph changed')
    return prep, summary

def import_map_validation(parts):
    """Fails when an owl:imports edge has no local ontology IRI or version IRI.

    Remote imports are never fetched. This is intentionally conservative; if
    a dependency exists only as an unrepresented remote import, do not claim
    full-context entailment verification.
    """
    known=set()
    requested=[]
    for name,g in parts.items():
        for s in g.subjects(RDF.type,OWL.Ontology):
            if isinstance(s,URIRef):
                known.add(str(s))
                for v in g.objects(s,OWL.versionIRI):
                    if isinstance(v,URIRef):known.add(str(v))
        for src,iri in g.subject_objects(OWL.imports):
            if isinstance(iri,URIRef):requested.append({'document':name,'iri':str(iri)})
    unresolved=[x for x in requested if x['iri'] not in known]
    return {'known_local_ontology_iris':sorted(known), 'declared_imports':requested,
            'unresolved_imports':unresolved}

def build_full_graph(parts,fold_idx,anchors=ANCHORS):
    """RDF union for audit ONLY. Never include augmented beside refined/source.

    The fold's two targets are removed *after* union to ensure an imported
    duplicate assertion cannot silently remain in the audit training graph.
    Their removal is recorded and reported.
    """
    g=Graph()
    for name,part in parts.items():
        for triple in part:g.add(triple)
    before=len(g)
    removed=[]
    for child,parent in FOLDS[fold_idx-1]:
        triple=(URIRef(child),RDFS.subClassOf,URIRef(parent))
        removed.append({'child':child,'parent':parent,'present_in_source_union':triple in g})
        g.remove(triple)
    for child,parent in anchors:
        g.add((URIRef(child),RDFS.subClassOf,URIRef(parent)))
    # Erase multi-document ontology headers/import links ONLY in generated copy:
    # all explicit local imports have already been loaded, after mapping check.
    for s in list(g.subjects(RDF.type,OWL.Ontology)):g.remove((s,RDF.type,OWL.Ontology))
    for s,o in list(g.subject_objects(OWL.imports)):g.remove((s,OWL.imports,o))
    # Standalone audit identity; avoids the original source module's IRI.
    g.add((URIRef('urn:project6:v9:owl-entailment:fold:'+str(fold_idx)), RDF.type,OWL.Ontology))
    if set(FOLDS[fold_idx-1])&(named(g)|closure(named(g))):
        raise RuntimeError(f'Fold {fold_idx}: asserted or transitive held-out target leakage')
    return g,{'source_union_triples':before,'full_audit_triples':len(g),'removed_targets':removed,
              'named_subclass_pairs':len(named(g))}

def check_augment_exact(root,parts):
    aug=load(root/'generated/module_augmented.ttl')
    union=Graph()
    for rel in ('src/module.ttl','generated/rail_axioms_refined.ttl'):
        for t in parts[rel]:union.add(t)
    # Isomorphism comparison respects blank-node identity renaming.
    from rdflib.compare import isomorphic
    return {'augmented_triples':len(aug),'source_refined_union_triples':len(union),
            'isomorphic_to_augmented':isomorphic(aug,union)}

def prepare(root):
    prep, summary=check_provenance(root)
    parts={}
    for rel in ('src/module.ttl','generated/rail_axioms_refined.ttl',*IMPORTS):
        log('Parsing '+rel)
        parts[rel]=load(root/rel)
    mapping=import_map_validation(parts)
    if mapping['unresolved_imports']:
        raise RuntimeError('Unmapped owl:imports: '+json.dumps(mapping['unresolved_imports'],indent=2)+
                           '\nProvide local matching ontology definitions; no remote imports can be silently omitted.')
    augment=check_augment_exact(root,parts)
    if not augment['isomorphic_to_augmented']:
        raise RuntimeError('augmented module is not an exact RDF union of source + refined; audit needs reviewed reconstruction')
    out=root/OUT; gen=root/GEN
    info={'version':VERSION,'status':'PREPARED_NOT_REASONED','contexts':{
        'named_v9':'exact v9 fold named subclass training ontology (what mOWL actually saw)',
        'full_context':'source + refined rail TBox + all six local imports + approved anchors - heldouts; NOT identical to mOWL training'
        },'original_v9_unmodified':True,'owl_entailment_checked':False,
        'source_hashes':summary['input_sha256'],'approved_anchor_count':7,
        'ontology_import_mapping':mapping,'source_augmented_crosscheck':augment,'folds':[]}
    for i, holdout in enumerate(FOLDS,1):
        named_file=root/prep['folds'][i-1]['outer_training_path']
        named_graph=load(named_file)
        expected=set(FOLDS[i-1]); edges=named(named_graph)
        if expected&(edges|closure(edges)):
            raise RuntimeError(f'v9 named graph fold {i} contains target')
        if not ANCHORS.issubset(edges):
            raise RuntimeError(f'Fold {i} missing approved anchor in original v9 named graph')
        if len(edges)!=prep['folds'][i-1]['outer_training_pairs']:
            raise RuntimeError(f'Fold {i} v9 named pair count changed')
        full, meta=build_full_graph(parts,i)
        missing_train=edges-named(full)
        if missing_train:
            raise RuntimeError(f'Fold {i} full context omitted {len(missing_train)} v9 train subclass axioms: '+repr(sorted(missing_train)[:5]))
        for ctx, graph in (('named_v9',named_graph),('full_context',full)):
            dest=gen/f'fold_{i}_{ctx}.owl'
            dest.parent.mkdir(parents=True,exist_ok=True)
            graph.serialize(destination=str(dest),format='xml')
            # Round-trip serialized OWL-document RDF input and repeat named checks.
            rt=Graph().parse(str(dest),format='xml')
            if expected&(named(rt)|closure(named(rt))):
                raise RuntimeError('RDF/XML roundtrip exposed target leakage')
        info['folds'].append({'fold':i,'holdouts':[{'child':c,'parent':p} for c,p in holdout],
            'named_v9':{'file':str((gen/f'fold_{i}_named_v9.owl').relative_to(root)).replace('\\','/'),
                        'triples':len(named_graph),'named_pairs':len(edges)},
            'full_context':{'file':str((gen/f'fold_{i}_full_context.owl').relative_to(root)).replace('\\','/'),**meta}})
    # hashes of generated input documents guard against edit before later reasoning
    for x in info['folds']:
        for ctx in ('named_v9','full_context'):
            x[ctx]['sha256']=digest(root/x[ctx]['file'])
    write_json(out/'preparation.json',info)
    return info

def reason_with_hermit(path,holdout,jvm_gb):
    """Exact OWLAPI isEntailed checks; no approximation by inferred direct edges."""
    import mowl
    import jpype
    if not jpype.isJVMStarted():mowl.init_jvm(f'{jvm_gb}g')
    from java.io import File
    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.owlapi.model import IRI
    try:
        factory=jpype.JClass('org.semanticweb.HermiT.ReasonerFactory')()
    except (TypeError,ImportError,AttributeError) as error:
        try:factory=jpype.JClass('org.semanticweb.HermiT.Reasoner$ReasonerFactory')()
        except Exception as second:
            raise RuntimeError('HermiT unavailable in the installed JVM. An ELK-only result is NOT a full OWL entailment proof.') from second
    mgr=OWLManager.createOWLOntologyManager()
    ontology=mgr.loadOntologyFromOntologyDocument(File(str(path.resolve())))
    imports_missing=[]
    for oi in ontology.getImportsDeclarations():imports_missing.append(str(oi.getIRI()))
    if imports_missing:raise RuntimeError('Audit input unexpectedly has unresolved OWL imports: '+str(imports_missing))
    data=mgr.getOWLDataFactory()
    signature={str(c.getIRI()) for c in ontology.getClassesInSignature()}
    for child,parent in holdout:
        if child not in signature or parent not in signature:
            raise RuntimeError(f'Missing holdout class signature after OWLAPI load: {child if child not in signature else parent}')
    r=factory.createReasoner(ontology)
    try:
        consistent=bool(r.isConsistent())
        rows=[]
        for child,parent in holdout:
            owl_child=data.getOWLClass(IRI.create(child))
            owl_parent=data.getOWLClass(IRI.create(parent))
            satisfiable=bool(r.isSatisfiable(owl_child)) if consistent else False
            entailed=bool(r.isEntailed(data.getOWLSubClassOfAxiom(owl_child,owl_parent))) if consistent else None
            rows.append({'child':child,'parent':parent,'entailed':entailed,
                         'child_satisfiable':satisfiable,
                         'status':('INCONSISTENT' if not consistent else 'UNSATISFIABLE_CHILD' if not satisfiable else
                                   'LEAKED_ENTAILED' if entailed else 'NOT_ENTAILED')})
        return {'consistent':consistent,'owlapi_logical_axiom_count':int(ontology.getLogicalAxiomCount()),
                'reasoner':'HermiT','results':rows}
    finally:r.dispose()

def run(root,info,jvm_gb):
    report={'version':VERSION,'status':'INCOMPLETE','full_owl_semantic_entailment_checked':False,
            'contexts':info['contexts'],'reasoner':'HermiT','folds':[]}
    output=root/OUT/'entailment_report.json'
    try:
        for fold in info['folds']:
            row={'fold':fold['fold'],'contexts':{}}
            for ctx in ('named_v9','full_context'):
                rel=fold[ctx]['file']
                if digest(root/rel)!=fold[ctx]['sha256']:
                    raise RuntimeError(f'Prepared OWL audit input was modified: {rel}')
                log(f'Fold {fold["fold"]}/4: HermiT {ctx}')
                targets=[(x['child'],x['parent']) for x in fold['holdouts']]
                row['contexts'][ctx]=reason_with_hermit(root/rel,targets,jvm_gb)
            report['folds'].append(row)
            write_json(output,report)
        bad=[]
        for fold in report['folds']:
            for ctx,rv in fold['contexts'].items():
                if not rv['consistent']:bad.append(f'Fold {fold["fold"]} {ctx}: inconsistent')
                for r in rv['results']:
                    if r['status']!='NOT_ENTAILED':bad.append(f'Fold {fold["fold"]} {ctx}: {r["child"]} -> {r["parent"]}: {r["status"]}')
        report['full_owl_semantic_entailment_checked']=True
        report['violations']=bad
        report['status']='PASS_HERMIT_BOTH_CONTEXTS' if not bad else 'FAIL_ENTAILMENT_OR_UNSATISFIABLE'
    except Exception as e:
        report['status']='ERROR_INCOMPLETE_DO_NOT_CLAIM_VALIDATED'
        report['error']=str(e)
        write_json(output,report)
        raise
    write_json(output,report)
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--jvm-gb',type=int,default=8)
    args=ap.parse_args()
    root=args.project_root.resolve()
    info=prepare(root)
    log('Prepared both OWL contexts with provenance and named-path checks PASS')
    if args.prepare_only:
        log('PREPARED ONLY; no OWL reasoner invoked')
        log(str(root/OUT/'preparation.json'))
        return 0
    result=run(root,info,args.jvm_gb)
    log('Result: '+result['status'])
    log(str(root/OUT/'entailment_report.json'))
    return 0 if result['status']=='PASS_HERMIT_BOTH_CONTEXTS' else 2

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as error:
        print('[v9 OWL audit] FAILED: '+str(error),file=sys.stderr,flush=True)
        sys.exit(2)
