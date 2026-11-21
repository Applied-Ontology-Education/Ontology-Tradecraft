#!/usr/bin/env python3
"""Prepare ROBOT module seeds and assemble a self-contained ontology.

All source triples except owl:imports are preserved; only imported vocabulary is trimmed.
"""
import argparse
from collections import Counter
from pathlib import Path
from rdflib import Graph, URIRef, BNode
from rdflib.compare import graph_diff, to_isomorphic
from rdflib.namespace import OWL, RDF, RDFS, XSD

STANDARD_PREFIXES = (
    str(OWL), str(RDF), str(RDFS), str(XSD),
    'http://www.w3.org/XML/1998/namespace',
)


def load(path):
    g = Graph()
    g.parse(str(path), format='turtle' if str(path).lower().endswith('.ttl') else None)
    return g


def prepare(args):
    source = load(args.input)
    imports = set(source.objects(None, OWL.imports))
    if not imports:
        raise SystemExit('No owl:imports found. This workflow targets imported ontology clutter.')
    # Every source term is a potential seed. This protects even UNUSED authored
    # classes, properties and individuals, plus imported vocabulary referenced
    # by authored axioms, labels, restrictions, domains, ranges, and GCAs.
    terms = set()
    for s, p, o in source:
        for term in (s, p, o):
            if not isinstance(term, URIRef):
                continue
            if str(term).startswith(STANDARD_PREFIXES):
                continue
            if term in imports:
                continue
            # Ontology metadata IRIs are not OWL entities; harmless if excluded.
            if term in set(source.subjects(RDF.type, OWL.Ontology)):
                continue
            terms.add(term)
    out = Path(args.seeds)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(f'<{str(t)}>\n' for t in sorted(terms, key=str)), encoding='utf-8')
    print(f'Source: {len(source)} triples; imports: {len(imports)}; module seeds: {len(terms)}')
    for i in sorted(imports, key=str): print(f'  import: {i}')
    print(f'Seed file: {out}')


def finish(args):
    original = load(args.input)
    imported_module = load(args.module)
    imports = set(original.triples((None, OWL.imports, None)))
    # Preserve ALL source data, including isolated user-created entities,
    # named individuals, custom GCAs, local classes, and property declarations.
    source_without_imports = Graph()
    for prefix, ns in original.namespaces(): source_without_imports.bind(prefix, ns)
    for t in original:
        if t not in imports:
            source_without_imports.add(t)
    combined = source_without_imports + imported_module
    # Remove all import declarations, including any from a module (a genuinely
    # standalone subset must not trigger Protege to reload full ontologies).
    for t in list(combined.triples((None, OWL.imports, None))): combined.remove(t)
    # Imported modules may include their own ontology headers: remove these
    # to keep the original ontology as the only ontology being edited.
    source_ontologies = set(original.subjects(RDF.type, OWL.Ontology))
    for ontology in set(imported_module.subjects(RDF.type, OWL.Ontology)) - source_ontologies:
        for t in list(combined.triples((ontology, None, None))): combined.remove(t)
    # RDF blank nodes will be canonicalized by the comparison implementation.
    in_a, only_source, only_result = graph_diff(to_isomorphic(source_without_imports), to_isomorphic(combined))
    if len(only_source):
        raise SystemExit(f'VALIDATION FAILED: {len(only_source)} original RDF triples absent from result; no output written.')
    if any(True for _ in combined.triples((None, OWL.imports, None))):
        raise SystemExit('VALIDATION FAILED: output still contains import declarations.')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.serialize(destination=str(output), format='turtle')
    loaded = load(output)
    # Verify round-trip serialization, including anonymous restrictions.
    _, missing, _ = graph_diff(to_isomorphic(combined), to_isomorphic(loaded))
    if len(missing):
        output.unlink(missing_ok=True)
        raise SystemExit('VALIDATION FAILED: serialized result lost RDF triples.')
    report = f'''ONTOLOGY IMPORT CLEANUP REPORT
Source: {args.input}
Original triples: {len(original)}
Imported module triples (before union): {len(imported_module)}
Combined standalone triples: {len(combined)}
Original source triples retained (except owl:imports): {len(source_without_imports)} / {len(original)-len(imports)}
Original owl:imports removed: {len(imports)}
Remaining owl:imports: 0

All user-authored classes, subclasses, property declarations, individual assertions,
and anonymous axioms (including GCAs) are preserved because ALL non-import
source triples are retained. The external ontology module is computed by ROBOT.

WARNING: Valid Turtle + preserved RDF is NOT an OWL consistency/entailment proof.
Run HermiT in Protege to check class inferences and ontology consistency.

NOTE: If the import closure was not fully resolved during ROBOT merge, this
output may have incomplete external axioms. Confirm imports were resolved,
and use a local ROBOT XML catalog (-CatalogFile) when needed.
'''
    Path(args.report).write_text(report, encoding='utf-8')
    print(report)
    print(f'Output: {output}')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--input', required=True); a.add_argument('--seeds', required=True)
    b = sub.add_parser('finish'); b.add_argument('--input', required=True); b.add_argument('--module', required=True); b.add_argument('--output', required=True); b.add_argument('--report', required=True)
    args = p.parse_args()
    (prepare if args.command == 'prepare' else finish)(args)

if __name__ == '__main__': main()
