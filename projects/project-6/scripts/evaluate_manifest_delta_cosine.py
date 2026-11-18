#!/usr/bin/env python3
"""Project 6: cosine audit for NEW Excel-ingested ABox only.

Compare the post-rail TBox baseline to the final enriched ontology. New
individual class assertions, object-property assertions, and data-property
assertions are the *only* scored items. Existing TBox axioms and rail-class
subclass pairs are never candidates.

The cosine measure is a lexical-semantic plausibility heuristic, NOT OWL
entailment, identity verification, datatype validation or independent truth.
The separate SHACL/HermiT stage checks formal constraints.

No mOWL training, Java, Grok, or learned vectors from the original TBox needed.
A sentence-transformers model is loaded locally for inference (first use may
require downloading public model weights); use --audit-only before installation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable

from rdflib import Graph, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS

VERSION = 'project6-scoped-abox-cosine-v1'
STATUS = 'PASS_SCOPED_DELTA_COSINE_EVALUATED'
DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_REVISION = '97ef89b0452be49c401bb722c776152b035909e7'
META_TYPES = {OWL.NamedIndividual, OWL.Class, OWL.ObjectProperty,
              OWL.DatatypeProperty, OWL.AnnotationProperty, OWL.Ontology,
              RDF.Property, RDFS.Class, OWL.Restriction}
ANNOTATIONS = {RDFS.label, RDFS.comment, RDFS.seeAlso, RDFS.isDefinedBy,
               OWL.versionInfo, SKOS.definition}
DEF_PROPS = [SKOS.definition,
             URIRef('http://purl.obolibrary.org/obo/IAO_0000115'),
             URIRef('http://www.w3.org/2004/02/skos/core#definition')]
SUPPORTED_IMPORTS = {'.ttl':'turtle', '.owl':None, '.rdf':None, '.xml':'xml',
                     '.nt':'nt', '.n3':'n3', '.jsonld':'json-ld', '.trig':'trig'}


def sha_file(p: Path) -> str:
    d = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            d.update(chunk)
    return d.hexdigest()


def plain_name(resource: URIRef | str) -> str:
    s = str(resource)
    s = s.rsplit('#',1)[-1].rsplit('/',1)[-1]
    return re.sub(r'[_-]+', ' ', s).strip()


def graph_file(p: Path) -> Graph:
    if not p.is_file():
        raise FileNotFoundError('Required graph missing: '+str(p))
    return Graph().parse(p, format='turtle' if p.suffix.lower()=='.ttl' else None)


def vocabulary(base: Graph, final: Graph, imports_dir: Path | None = None) -> tuple[Graph,dict]:
    graph = Graph()
    graph += base
    graph += final
    loaded=[]; errors=[]
    if imports_dir and imports_dir.is_dir():
        for p in sorted(imports_dir.iterdir(),key=lambda p:p.name):
            if p.suffix.lower() not in SUPPORTED_IMPORTS or not p.is_file():
                continue
            try:
                graph.parse(p,format=SUPPORTED_IMPORTS[p.suffix.lower()])
                loaded.append(p.name)
            except Exception as exc:
                errors.append({'file':p.name,'error':str(exc)[:200]})
    return graph, {'loaded_local_imports':loaded,'failed_local_imports':errors}


def label(graph: Graph, iri: URIRef) -> str:
    values=[str(v) for v in graph.objects(iri,RDFS.label) if isinstance(v,Literal)]
    return sorted(values, key=lambda v:(len(v),v))[0] if values else plain_name(iri)


def definition(graph: Graph, iri: URIRef) -> str:
    vals=[]
    for p in DEF_PROPS:
        vals += [str(v) for v in graph.objects(iri,p) if isinstance(v,Literal)]
    return sorted(set(vals),key=lambda v:(len(v),v))[0][:400] if vals else ''


def description(graph: Graph, iri: URIRef) -> str:
    return ' '.join(s for s in [label(graph, iri),definition(graph,iri)] if s)


def individual_types(graph: Graph, iri: URIRef) -> list[URIRef]:
    return sorted({c for c in graph.objects(iri,RDF.type)
                   if isinstance(c,URIRef) and c not in META_TYPES},key=str)


def individual_label(graph: Graph, iri: URIRef) -> str:
    labels=[str(x) for x in graph.objects(iri,RDFS.label) if isinstance(x,Literal)]
    # Do not fall back to a hashed manifest identifier as semantic evidence.
    return sorted(labels,key=lambda s:(len(s),s))[0] if labels else ''


def value_kind(literal: Literal) -> str:
    dt=literal.datatype
    if dt in {XSD.dateTime,XSD.date,XSD.time}:
        return 'a time or date value'
    if dt in {XSD.decimal,XSD.integer,XSD.float,XSD.double,XSD.int,XSD.nonNegativeInteger}:
        return 'a numeric value'
    if dt==XSD.boolean:
        return 'a true or false value'
    s=str(literal)
    if re.fullmatch(r'\d{4}-\d\d-\d\d(?:T\S+)?',s):
        return 'a time or date value'
    if re.fullmatch(r'[A-Z0-9_-]{2,}',s,re.I) and any(ch.isdigit() for ch in s):
        return 'an alphanumeric identifier'
    if len(s.split())>=2:
        return 'a multiword text value'
    return 'a short text value'


def token(terms: tuple) -> str:
    raw='\u001e'.join(term.n3() for term in terms)
    return hashlib.sha256(raw.encode('utf8')).hexdigest()[:20]


def _is_original_subject(s: URIRef, orig_subjects: set[URIRef]) -> bool:
    return s in orig_subjects


def select_new_assertions(base: Graph, final: Graph, schema: Graph,
                          vocab: Graph) -> dict[str,Any]:
    """Select every new scored ABox assertion; never select TBox or imports."""
    original_subjects={s for s in base.subjects() if isinstance(s, URIRef)}
    all_individuals={s for s in final.subjects(RDF.type, OWL.NamedIndividual)
                     if isinstance(s,URIRef)}
    # Account for graphs where individual declarations omit owl:NamedIndividual.
    known_types={s for s,p,o in final.triples((None,RDF.type,None))
                 if isinstance(s,URIRef) and isinstance(o,URIRef)
                 and o not in META_TYPES and s not in {OWL.Thing,OWL.Nothing}}
    all_individuals |= known_types
    new_individuals=all_individuals-original_subjects
    obj_props={s for s in vocab.subjects(RDF.type,OWL.ObjectProperty)
               if isinstance(s,URIRef)}
    data_props={s for s in vocab.subjects(RDF.type,OWL.DatatypeProperty)
                if isinstance(s,URIRef)}
    new_tbox_obj={s for s in final.subjects(RDF.type, OWL.ObjectProperty)
                  if isinstance(s,URIRef) and (s,RDF.type,OWL.ObjectProperty) not in base}
    new_tbox_data={s for s in final.subjects(RDF.type, OWL.DatatypeProperty)
                   if isinstance(s,URIRef) and (s,RDF.type,OWL.DatatypeProperty) not in base}
    categorized={k:[] for k in ['individual_class_assertion','object_property_assertion','data_property_assertion']}
    exclusions=Counter(); unresolved=[]
    delta=[]
    for s,p,o in final:
        if isinstance(s,URIRef) and (s,p,o) not in base:
            delta.append((s,p,o))
    for s,p,o in sorted(delta,key=lambda t:tuple(str(v) for v in t)):
        # New property declarations, class axioms, and annotations are never scored.
        if p==RDF.type and o in META_TYPES:
            exclusions['tbox_or_individual_declarations']+=1;continue
        if p in ANNOTATIONS:
            exclusions['annotations']+=1;continue
        if s not in all_individuals:
            exclusions['non_individual_schema_statements']+=1;continue
        if p==RDF.type and isinstance(o,URIRef):
            categorized['individual_class_assertion'].append((s,p,o));continue
        if isinstance(o,URIRef) and (p in obj_props or p not in data_props):
            categorized['object_property_assertion'].append((s,p,o))
            if p not in obj_props:unresolved.append({'id':token((s,p,o)), 'kind':'undeclared_object_property','predicate':str(p)})
            continue
        if isinstance(o,Literal) and (p in data_props or p not in obj_props):
            categorized['data_property_assertion'].append((s,p,o))
            if p not in data_props:unresolved.append({'id':token((s,p,o)), 'kind':'undeclared_data_property','predicate':str(p)})
            continue
        unresolved.append({'id':token((s,p,o)), 'kind':'invalid_property_object_kind','predicate':str(p)})
    return {'selected':categorized,
            'counts':{'new_individuals':len(new_individuals),'baseline_triples':len(base),
                      'final_triples':len(final),'new_named_object_properties':len(new_tbox_obj),
                      'new_named_data_properties':len(new_tbox_data),
                      'new_named_classes':len({s for s in final.subjects(RDF.type,OWL.Class)
                                               if isinstance(s,URIRef) and (s,RDF.type,OWL.Class) not in base}),
                      'new_named_individuals_declared':len(new_individuals),
                      'new_triples_with_named_subjects':len(delta),
                      'excluded_new_triples':dict(exclusions)},
            'unresolved':unresolved}


def comparisons(selected: dict[str,list[tuple]], final: Graph, vocab: Graph) -> list[dict[str,Any]]:
    rows=[]
    for category, triples in selected.items():
        for s,p,o in triples:
            subject_label=individual_label(final,s)
            types=individual_types(final,s)
            type_text=', '.join(label(vocab,c) for c in types)
            flag=''
            if category=='individual_class_assertion':
                target=description(vocab,o)
                # Ingestion-generated labels can literally say "rail ticket" while
                # the asserted type is RailTicket. Remove the type words so cosine
                # is not won by simply copying the target into the evidence.
                type_label=label(vocab,o)
                redacted_label=subject_label
                if subject_label and type_label and type_label.casefold() in subject_label.casefold():
                    redacted_label=re.sub(re.escape(type_label), '[type omitted]',
                                          subject_label,flags=re.IGNORECASE)
                    flag='generated_instance_label_repeats_type_removed'
                evidence=f'Manifest entity description: {redacted_label}.'
                other_props=[label(vocab,q) for q in set(final.predicates(s,None))
                             if q not in ANNOTATIONS and q!=RDF.type]
                # Other observed property names may supply context; target type not repeated here.
                if other_props:evidence+=' Recorded fields and links: '+', '.join(sorted(other_props))+'.'
                if not redacted_label and not other_props:flag='insufficient_independent_context'
                target_iri=str(o)
            elif category=='object_property_assertion':
                object_label=individual_label(final,o) if isinstance(o,URIRef) else ''
                obj_types=individual_types(final,o) if isinstance(o,URIRef) else []
                object_type_text=', '.join(label(vocab,t) for t in obj_types)
                evidence=('An entity of type '+(type_text or 'unknown type')+
                          (' ('+subject_label+')' if subject_label else '')+
                          ' is related to an entity of type '+(object_type_text or 'unknown type')+
                          (' ('+object_label+')' if object_label else '')+'.')
                # The predicate being evaluated is not copied into its evidence text.
                target=description(vocab,p)
                target_iri=str(p)
                if not obj_types or not types: flag='weak_endpoint_type_context'
            else:
                evidence=('A manifest entity of type '+(type_text or 'unknown type')+
                          (' ('+subject_label+')' if subject_label else '')+
                          ' records '+value_kind(o)+'.')
                target=description(vocab,p)
                target_iri=str(p)
                if not types:flag='weak_subject_type_context'
            rows.append({'id':token((s,p,o)),'category':category,
                         'predicate':str(p), 'target':target_iri,
                         'evidence':evidence,'reference':target,'review_flag':flag})
    return rows


def compute_scores(rows: list[dict], embedder: Callable[[list[str]],Any]) -> list[dict]:
    if not rows:raise ValueError('No new ABox assertions to evaluate')
    texts=[]
    for row in rows:
        if not row['evidence'].strip() or not row['reference'].strip():
            raise ValueError('Missing evidence/reference for '+row['id'])
        texts.extend([row['evidence'],row['reference']])
    vectors=embedder(texts)
    import numpy as np
    a=np.asarray(vectors,dtype='float64')
    if a.ndim!=2 or a.shape[0]!=2*len(rows):
        raise ValueError('Embedding model returned invalid vector shape')
    n=np.linalg.norm(a,axis=1)
    if not np.all(np.isfinite(a)) or np.any(n < 1e-12):
        raise ValueError('Invalid/nonfinite/zero embedding vectors')
    a=a/n[:,None]
    scored=[]
    for i,row in enumerate(rows):
        score=float(np.clip(a[2*i].dot(a[2*i+1]),-1,1))
        scored.append({'id':row['id'], 'category':row['category'],
                       'predicate':row['predicate'], 'target':row['target'],
                       'cosine':round(score,6),'review_flag':row['review_flag']})
    return scored


def summarize(scored: list[dict], threshold: float) -> dict[str,Any]:
    groups=defaultdict(list)
    for row in scored:groups[row['category']].append(row['cosine'])
    stats={}
    for name in ['individual_class_assertion','object_property_assertion','data_property_assertion']:
        values=groups[name]
        stats[name]={'count':len(values), 'mean':round(sum(values)/len(values),6) if values else None,
                     'min':min(values) if values else None, 'max':max(values) if values else None,
                     'at_or_above_threshold':sum(v>=threshold for v in values)}
    allvals=[v for vs in groups.values() for v in vs]
    mean=sum(allvals)/len(allvals) if allvals else None
    macro=[v['mean'] for v in stats.values() if v['mean'] is not None]
    return {'total_scored_assertions':len(allvals),'mean_cosine':round(mean,6) if mean is not None else None,
            'macro_category_mean':round(sum(macro)/len(macro),6) if macro else None,
            'threshold':threshold,'metric_threshold_met':mean is not None and mean>=threshold,
            'by_category':stats}


def run(root: Path, *, baseline: Path | None = None, final: Path | None = None,
        schema: Path | None = None, imports_dir: Path | None = None,
        model: str = DEFAULT_MODEL, revision: str | None = DEFAULT_REVISION,
        threshold: float = .70,
        audit_only: bool = False, local_files_only: bool = False,
        embedder: Callable[[list[str]],Any] | None = None,
        report_path: Path | None = None) -> dict:
    root=root.resolve()
    baseline=baseline or (root/'generated/module_augmented.ttl')
    final=final or (root/'generated/module_augmented_with_manifest_enriched.ttl')
    schema=schema or (root/'generated/manifest_tbox_extension.ttl')
    imports_dir=imports_dir if imports_dir is not None else (root/'imports')
    report_path=report_path or (root/'reports/manifest_delta_cosine.json')
    if not 0<=threshold<=1:
        raise ValueError('--threshold must be between 0 and 1')
    for p in [baseline,final,schema]:
        if not p.is_file():raise FileNotFoundError('Missing required pipeline output: '+str(p))
    if not final.read_bytes().startswith(baseline.read_bytes()):
        raise ValueError('Final ontology does not retain exact augmented TBox prefix; refusing delta evaluation')
    print('[ABox cosine] Parsing augmented baseline and enriched final ontology',flush=True)
    g0=graph_file(baseline);gf=graph_file(final);gs=graph_file(schema)
    vocab,import_info=vocabulary(g0,gf,imports_dir)
    chosen=select_new_assertions(g0,gf,gs,vocab)
    candidates=comparisons(chosen['selected'],gf,vocab)
    counts={k:len(v) for k,v in chosen['selected'].items()}
    print('[ABox cosine] Scoped assertion counts:',counts,flush=True)
    if not candidates:
        raise RuntimeError('No new individual / object / data property assertions identified')
    # Do not silently discard valid ABox usages of an inherited property whose
    # declaration lives outside the local six-import vocabulary. Report them
    # explicitly; the formal SHACL/HermiT stage makes the validity decision.
    report={'version':VERSION,'status':'PASS_SCOPED_DELTA_AUDIT_ONLY' if audit_only else STATUS,
            'scope':'NEW_EXCEL_ABOX_ASSERTIONS_ONLY',
            'score_definition':'Cosine of evidence-context and ontology-target text embeddings; semantic heuristic, not OWL correctness.',
            'source_of_new_assertions':'final minus augmented baseline, restricted to OWL named individual assertions and property usage',
            'original_classes_scored':0, 'original_subclass_axioms_scored':0,
            'rail_tbox_subclass_axioms_scored':0,
            'model':None if audit_only else model,
            'model_revision':None if audit_only else revision,
            'source_sha256':sha_file(root/'src/module.ttl') if (root/'src/module.ttl').is_file() else None,
            'baseline_sha256':sha_file(baseline), 'final_sha256':sha_file(final),
            'manifest_schema_sha256':sha_file(schema),
            'imports':import_info,'counts':chosen['counts'],
            'selected_assertions':counts,'total_selected_assertions':len(candidates),
            'unresolved_property_usages':chosen['unresolved'],
            'warnings':[
             'Generated instance labels may echo their target type; exact type phrases are removed before embedding.',
             'Object/data-property comparisons use assertion endpoint types and value kinds, not independently adjudicated factual evidence.',
             'Raw literal identifiers, names and dates are intentionally excluded from the JSON report.',
             'Cosine is not an OWL correctness test; use HermiT, SHACL and workbook evidence checks separately.',
             'The 0.70 threshold has not been calibrated as a semantic-validity decision boundary.'
            ]}
    if not audit_only:
        if embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError('Install sentence-transformers in .venv-mowl: python -m pip install sentence-transformers') from exc
            print('[ABox cosine] Loading pretrained sentence model (no training or Java): '+model,flush=True)
            try:
                st=SentenceTransformer(model,device='cpu',revision=revision,
                                       local_files_only=local_files_only)
            except Exception as exc:
                raise RuntimeError('Could not load sentence model; check the model download/cache: '+str(exc)) from exc
            embedder=lambda texts:st.encode(texts,batch_size=32,normalize_embeddings=True,
                                             convert_to_numpy=True,show_progress_bar=False)
        print('[ABox cosine] Embedding '+str(len(candidates)*2)+' texts, computing cosine',flush=True)
        scored=compute_scores(candidates,embedder)
        report['results']=summarize(scored,threshold)
        report['assertions']=scored
        print('[ABox cosine] Mean:',report['results']['mean_cosine'],
              'passes 0.70 target:',report['results']['metric_threshold_met'],flush=True)
    else:
        report['audit_only']=True
        report['results']=None
        print('[ABox cosine] Audit completed without loading an embedding model',flush=True)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    tmp=report_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf8')
    tmp.replace(report_path)
    print('[ABox cosine] Report:',report_path,flush=True)
    return report


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--baseline',type=Path)
    p.add_argument('--final',type=Path)
    p.add_argument('--schema',type=Path)
    p.add_argument('--imports-dir',type=Path)
    p.add_argument('--model',default=DEFAULT_MODEL)
    p.add_argument('--revision',default=DEFAULT_REVISION,
                   help='Pinned embedding model snapshot; default is a fixed public model commit')
    p.add_argument('--threshold',type=float,default=.70)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--local-files-only',action='store_true',help='Do not download model weights')
    p.add_argument('--enforce-threshold',action='store_true',help='Exit 2 when mean cosine is below threshold')
    p.add_argument('--report',type=Path)
    args=p.parse_args(argv)
    try:
        output=run(args.project_root,baseline=args.baseline,final=args.final,
                   schema=args.schema,imports_dir=args.imports_dir,
                   model=args.model,revision=args.revision,threshold=args.threshold,audit_only=args.audit_only,
                   local_files_only=args.local_files_only,report_path=args.report)
        if args.enforce_threshold and not args.audit_only and not output['results']['metric_threshold_met']:
            print('COSINE BELOW TARGET (report saved)',file=sys.stderr)
            return 2
        return 0
    except Exception as exc:
        print('ABOX COSINE ERROR: '+str(exc),file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
