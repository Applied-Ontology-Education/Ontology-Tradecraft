"""Project 6 experimental Step 5A audit of independent railway superclass evidence.

Run from projects/project-6 root (with the existing experiment_rail_mowl_v8.py
in scripts/):

    python -u scripts/audit_rail_superclass_support.py
    python -u scripts/audit_rail_superclass_support.py --reviewed-anchors PATH_TO_JSON

Read-only upstream inputs; outputs under reports/experiments/rail_anchor_review
and (only when reviewed anchors supplied) generated/experiments/rail_anchor_review.

Deliberate limitations:
 * Neither textual similarity nor property domain/range creates subclass axioms.
 * Named-graph reachability checks are not full OWL DL entailment checks.
 * No candidate is generated from a held-out edge's ancestor chain.
 * Adding reviewed anchors changes the evaluation setup and needs a fresh,
   independent reasoner and evaluation review before any performance claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdflib import BNode, Graph, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS, SKOS

import experiment_rail_mowl_v8 as v8

VERSION = 'project6-experimental-superclass-support-review-v1'
REPORT_DIR = Path('reports/experiments/rail_anchor_review')
GRAPH_DIR = Path('generated/experiments/rail_anchor_review')
RAIL_CHILDREN = {c for fold in v8.FOLDS for c, _ in fold}
ALLOWED_EXTENSIONS = {'.ttl': 'turtle', '.nt': 'nt', '.n3': 'n3',
                      '.rdf': 'xml', '.xml': 'xml', '.owl': None}
ANNOTATION_PROPERTIES = (RDFS.label, SKOS.definition, RDFS.comment,
                         URIRef('http://purl.obolibrary.org/obo/IAO_0000115'))


def log(msg: str) -> None:
    print('[Rail anchor audit] ' + msg, flush=True)


def parse_ontology(path: Path) -> Graph:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f'Unsupported ontology format: {path}')
    candidates = ['xml', 'turtle'] if suffix == '.owl' else [ALLOWED_EXTENSIONS[suffix]]
    errors = []
    for candidate in candidates:
        try:
            graph = Graph()
            graph.parse(path, format=candidate)
            return graph
        except Exception as exc:
            errors.append(f'{candidate}: {exc}')
    raise RuntimeError(f'Cannot parse {path}: ' + '; '.join(errors)[:500])


def inspect_sources(root: Path) -> tuple[dict[str, Graph], dict[str, str]]:
    files = [root/'src/module.ttl', root/'generated/module_augmented.ttl']
    refined = root/'generated/rail_axioms_refined.ttl'
    if refined.is_file():
        files.append(refined)
    import_dir = root/'imports'
    if import_dir.is_dir():
        files += sorted(p for p in import_dir.rglob('*')
                        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS)
    result: dict[str, Graph] = {}
    hashes = {}
    for path in files:
        rel = path.relative_to(root).as_posix()
        log(f'Reading ontology evidence: {rel}')
        hashes[rel] = v8.digest(path)
        result[rel] = parse_ontology(path)
    return result, hashes


def union_graph(graphs: dict[str, Graph]) -> Graph:
    result = Graph()
    for graph in graphs.values():
        result += graph
    return result


def structural_mentions(g: Graph, child: URIRef) -> list[dict]:
    """Inventory references, without incorrectly treating mentions as EL supervision."""
    result = []
    for prop in sorted(g.subjects(RDFS.domain, child), key=str):
        result.append({'kind': 'rdfs:domain', 'property': str(prop)})
    for prop in sorted(g.subjects(RDFS.range, child), key=str):
        result.append({'kind': 'rdfs:range', 'property': str(prop)})
    for predicate, name in ((OWL.someValuesFrom, 'owl:someValuesFrom'),
                            (OWL.allValuesFrom, 'owl:allValuesFrom'),
                            (OWL.hasValue, 'owl:hasValue')):
        for expr in sorted(g.subjects(predicate, child), key=str):
            result.append({'kind': name, 'expression_node': str(expr)})
    for predicate, name in ((OWL.equivalentClass, 'owl:equivalentClass'),
                            (OWL.disjointWith, 'owl:disjointWith')):
        for obj in sorted(g.objects(child, predicate), key=str):
            result.append({'kind': name, 'object': str(obj)})
        for obj in sorted(g.subjects(predicate, child), key=str):
            result.append({'kind': name + '_reverse', 'subject': str(obj)})
    for expr, node in g.subject_objects(OWL.intersectionOf):
        if not isinstance(node, BNode):
            continue
        try:
            if child in Collection(g, node):
                result.append({'kind': 'owl:intersectionOf_member',
                               'expression_node': str(expr)})
        except Exception:
            result.append({'kind': 'malformed_owl:intersectionOf',
                           'expression_node': str(expr)})
    return sorted(result, key=lambda x: json.dumps(x, sort_keys=True))


def document_annotations(g: Graph, child: URIRef) -> dict[str, list[str]]:
    return {str(prop): sorted({str(o) for o in g.objects(child, prop)})
            for prop in ANNOTATION_PROPERTIES if any(g.objects(child, prop))}


def evidence_audit(root: Path, inputs: dict, snapshot: dict[str, str]) -> dict:
    if v8.immutability_snapshot(root) != snapshot:
        raise RuntimeError('Upstream inputs changed before audit')
    graphs, hashes = inspect_sources(root)
    all_graph = union_graph(graphs)
    full_pairs = inputs['edges']
    existing_ancestors = v8.closure(full_pairs)
    source_details = {p: {'triples': len(g), 'sha256': hashes[p]}
                      for p, g in graphs.items()}
    fold_results = []
    total_status = Counter()
    for fold_idx, hold in enumerate(v8.FOLDS, start=1):
        training = full_pairs - set(hold)
        if set(hold) & (training | v8.closure(training)):
            raise RuntimeError(f'Fold {fold_idx}: leaked target')
        members = []
        for child, target in hold:
            subj = URIRef(child)
            state = v8.determine_support(training, child)
            total_status[state['support_status']] += 1
            retained = sorted(p for c, p in training if c == child)
            incoming = sorted(c for c, p in training if p == child)
            # IMPORTANT: anything reached in the *full* ontology only through
            # the held-out relationship is explicitly labelled target-dependent;
            # it is NEVER an automatic proposal for training.
            ancestors_without_target = {p for c, p in v8.closure(training) if c == child}
            ancestors_full = {p for c, p in existing_ancestors if c == child}
            dependent = sorted(ancestors_full - ancestors_without_target - {target})
            direct_full = sorted(p for c, p in full_pairs if c == child)
            candidates_existing = sorted(p for p in retained if p != target)
            per_file = {}
            for path, graph in graphs.items():
                mentions = structural_mentions(graph, subj)
                anns = document_annotations(graph, subj)
                if mentions or anns:
                    per_file[path] = {
                        'structural_references': mentions,
                        'annotations': anns,
                    }
            members.append({
                'child': child, 'withheld_target': target,
                **state,
                'retained_direct_superclasses': retained,
                'incoming_training_children': incoming,
                'existing_non_target_anchors': candidates_existing,
                'target_dependent_ancestors_NOT_independent_evidence': dependent,
                'full_graph_direct_superclasses_FOR_AUDIT_ONLY': direct_full,
                'other_owl_or_annotation_evidence_NOT_subclass_axioms': per_file,
                'independent_anchor_gap': len(retained) == 0,
            })
        fold_results.append({'fold': fold_idx, 'train_pairs': len(training),
                             'heldout': members})
    if v8.immutability_snapshot(root) != snapshot:
        raise RuntimeError('Upstream inputs changed during audit')
    return {
        'version': VERSION, 'status': 'SUPPORT_EVIDENCE_AUDITED_NO_AXIOMS_ADDED',
        'fold_count': 4, 'rail_classes': 8,
        'training_representation': 'existing named taxonomy only; other structural evidence is inventoried, not silently converted',
        'support_counts': dict(sorted(total_status.items())),
        'input_hashes': snapshot, 'ontology_evidence_sources': source_details,
        'folds': fold_results,
        'recommendation': 'Review potential independent superclass evidence; do not infer superclass edges from a held-out assertion or labels. OWL reasoner check required for new fold datasets.',
        'owl_semantic_leakage_proven_absent': False,
        'named_taxonomy_leakage_detected': False,
        'original_pipeline_modified': False,
    }


def review_template(report: dict) -> dict:
    return {
        'version': VERSION,
        '_instructions': ('OPTIONAL review file. Fill approved_anchors only with independently justified, '
                          'human-reviewed OWL subclass axioms; do not infer these from a withheld target. '
                          'Provide independent_source, reviewer, and rationale. Submit this file with '
                          '--reviewed-anchors; newly added evidence must undergo OWL entailment checking.'),
        'approved_anchors': [],
        'review_queue': [
            {'fold': fold['fold'], 'child': item['child'],
             'target_to_remain_held_out': item['withheld_target'],
             'retained_direct_superclasses': item['retained_direct_superclasses'],
             'requires_independent_ontology_justification': item['independent_anchor_gap']}
            for fold in report['folds'] for item in fold['heldout']
        ],
        'example_only_not_an_approval': {
            'child': v8.BASE + 'rail_ticket',
            'parent': 'http://purl.obolibrary.org/obo/IAO_0000178',
            'rationale': 'Example only: this must be justified independently of the withheld rail_ticket->ticket axiom',
            'independent_source': 'Replace with actual independently verified ontology evidence',
            'reviewer': 'Replace with actual reviewer ID or course review note',
        },
    }


def validated_anchors(path: Path, inputs: dict) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('version') != VERSION:
        raise ValueError('Review file must use this script version')
    items = data.get('approved_anchors')
    if not isinstance(items, list) or not items:
        raise ValueError('Review file must contain at least one approved_anchors item; no axiom fabricated')
    expected = inputs['expected']
    iri_inventory = inputs['classes'] | {i for pair in inputs['edges'] for i in pair}
    validated: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('Every approved anchor must be an object')
        child, parent = item.get('child'), item.get('parent')
        if not isinstance(child, str) or not isinstance(parent, str):
            raise ValueError('All approved anchors must have string child/parent IRIs')
        if not child.startswith('http') or not parent.startswith('http') or child == parent:
            raise ValueError('Approved anchors must have valid distinct HTTP(S) IRIs')
        if child not in RAIL_CHILDREN:
            raise ValueError(f'Only the eight audited railway children may gain anchors: {child}')
        if parent not in iri_inventory:
            raise ValueError(f'Unknown parent class; add reviewed ontology import/declaration before use: {parent}')
        for required in ('rationale', 'independent_source', 'reviewer'):
            if not isinstance(item.get(required), str) or len(item[required].strip()) < 10:
                raise ValueError(f'Anchor {child} missing substantive {required}')
        if (child, parent) in expected:
            raise ValueError('Cannot reintroduce held-out target edge as an anchor')
        if (child, parent) in inputs['edges']:
            raise ValueError('Anchor already exists: nothing new to add')
        validated.add((child, parent))
    if len(validated) != len(items):
        raise ValueError('Duplicate anchors in review file')
    for fold_idx, hold in enumerate(v8.FOLDS, start=1):
        training = (inputs['edges'] - set(hold)) | validated
        leakage = set(hold) & v8.closure(training)
        if leakage or (set(hold) & training):
            raise ValueError(f'Fold {fold_idx}: reviewed anchors create named superclass leakage: {sorted(leakage)}')
    return sorted(validated)


def write_reviewed_folds(root: Path, inputs: dict, report: dict,
                         anchors: list[tuple[str, str]], review_path: Path,
                         snapshot: dict[str, str]) -> dict:
    if v8.immutability_snapshot(root) != snapshot:
        raise RuntimeError('Upstream inputs changed before reviewed-fold creation')
    output = root / GRAPH_DIR
    output.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    for idx, hold in enumerate(v8.FOLDS, 1):
        original = inputs['edges'] - set(hold)
        edges = original | set(anchors)
        # Double-check after constructing complete fold graph.
        if set(hold) & (edges | v8.closure(edges)):
            raise RuntimeError(f'Leakage after fold creation: {idx}')
        path = output / f'fold_{idx}_train_reviewed.ttl'
        classes = inputs['classes'] | {term for e in anchors for term in e}
        v8.save_training_graph(path, edges, classes)
        reparsed = v8.named_pairs(v8.graph_load(path))
        if reparsed != edges:
            raise RuntimeError('Round-trip mismatch in reviewed fold')
        fold_rows.append({
            'fold': idx,
            'training_path': path.relative_to(root).as_posix(),
            'sha256': v8.digest(path),
            'training_pairs': len(edges),
            'held_out_targets': [{'child': c, 'parent': p} for c, p in hold],
            'support_after': [{**v8.determine_support(edges, c), 'child': c}
                              for c, _ in hold],
        })
    if v8.immutability_snapshot(root) != snapshot:
        raise RuntimeError('Upstream inputs changed during fold regeneration')
    saved = {
        'version': VERSION,
        'status': 'REVIEWED_FOLDS_PREPARED_REQUIRES_REASONER_ENTAILMENT_CHECK',
        'review_file_sha256': v8.digest(review_path),
        'approved_anchor_count': len(anchors),
        'approved_anchor_edges': [{'child': c, 'parent': p} for c, p in anchors],
        'named_taxonomy_holdouts_not_reachable': True,
        'full_owl_entailment_checked': False,
        'eligible_for_claims_of_leakage_free_owl_cv': False,
        'input_hashes': snapshot,
        'original_pipeline_modified': False,
        'folds': fold_rows,
    }
    v8.save_json(root / REPORT_DIR / 'reviewed_folds.json', saved)
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path,
                    default=Path(__file__).resolve().parents[1])
    ap.add_argument('--reviewed-anchors', type=Path, default=None,
                    help='Optional HUMAN reviewed superclass edges; never autofilled')
    args = ap.parse_args()
    root = args.project_root.resolve()
    try:
        snapshot = v8.immutability_snapshot(root)
        inputs = v8.load_split(root)  # verify the existing Step 5A hashes/rail holdouts
        audit = evidence_audit(root, inputs, snapshot)
        v8.save_json(root / REPORT_DIR / 'superclass_support_audit.json', audit)
        template = root / REPORT_DIR / 'anchor_review_template.json'
        # A reviewed template must never be overwritten by another audit run.
        if not template.is_file():
            v8.save_json(template, review_template(audit))
        log('SUPPORT AUDIT complete: ' + str(audit['support_counts']))
        log('Report: ' + str(root / REPORT_DIR / 'superclass_support_audit.json'))
        if args.reviewed_anchors:
            supplied = args.reviewed_anchors.resolve()
            anchors = validated_anchors(supplied, inputs)
            saved = write_reviewed_folds(root, inputs, audit, anchors, supplied, snapshot)
            log('Reviewed fold preparation: ' + saved['status'])
        else:
            log('No new superclass axioms added. Review evidence before creating enriched folds.')
        return 0
    except Exception as exc:
        print('RAIL ANCHOR AUDIT FAILED: ' + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
