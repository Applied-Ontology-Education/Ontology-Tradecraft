#!/usr/bin/env python3
"""Read-only audit for duplicate IAO ontology IRI and CCO version provenance.

Run from project-6 root:
 python -u scripts/diagnose_v9_ontology_identity.py

It does not edit the existing source, imports, training data, frozen v9, or
reasoner reports. Output is an isolated JSON diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import RDF, RDFS, OWL

FILES = (
    "imports/core.ttl",
    "imports/iao-edit.owl",
    "imports/InformationEntityOntology.ttl",
    "imports/AgentOntology.ttl",
    "imports/EventOntology.ttl",
)
OUT = "reports/experiments/rail_mowl_v9_owl_audit_v2/ontology_identity_diagnostic.json"
PREDICATES = {
    "named_subclass_axioms": RDFS.subClassOf,
    "equivalent_class_axioms": OWL.equivalentClass,
    "disjoint_class_axioms": OWL.disjointWith,
    "equivalent_property_axioms": OWL.equivalentProperty,
    "inverse_property_axioms": OWL.inverseOf,
    "object_property_domains": RDFS.domain,
    "object_property_ranges": RDFS.range,
    "property_chains": OWL.propertyChainAxiom,
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse(path: Path) -> Graph:
    if not path.is_file():
        raise FileNotFoundError(f"Required original import file not found: {path}")
    formats = ("xml", "turtle") if path.suffix.lower() == ".owl" else ("turtle", "xml")
    errors = []
    for fmt in formats:
        try:
            g = Graph()
            g.parse(str(path), format=fmt)
            return g
        except Exception as ex:
            errors.append(f"{fmt}: {str(ex)[:160]}")
    raise RuntimeError(f"Failed to parse {path}: {'; '.join(errors)}")


def inspect(path: Path, graph: Graph) -> dict:
    ontologies = sorted(str(iri) for iri in graph.subjects(RDF.type, OWL.Ontology) if isinstance(iri, URIRef))
    identities = []
    for iri in ontologies:
        versions = sorted(str(v) for v in graph.objects(URIRef(iri), OWL.versionIRI) if isinstance(v, URIRef))
        identities.append({"ontology_iri": iri, "version_iris": versions})
    return {
        "sha256": digest(path),
        "triple_count": len(graph),
        "ontology_identities": identities,
        "imports": sorted(str(iri) for iri in graph.objects(None, OWL.imports) if isinstance(iri, URIRef)),
        "axiom_indicators": {label: sum(1 for _ in graph.triples((None, predicate, None))) for label, predicate in PREDICATES.items()},
    }


def diagnose(root: Path) -> dict:
    graphs = {}
    records = {}
    for rel in FILES:
        print("[v9 identity] Reading", rel, flush=True)
        p = root / rel
        g = parse(p)
        graphs[rel] = g
        records[rel] = inspect(p, g)
    first, second = FILES[:2]
    first_ids = {row["ontology_iri"] for row in records[first]["ontology_identities"]}
    second_ids = {row["ontology_iri"] for row in records[second]["ontology_identities"]}
    common = sorted(first_ids & second_ids)
    print("[v9 identity] Comparing RDF graph equivalence for duplicate IAO identities", flush=True)
    identical = isomorphic(graphs[first], graphs[second])
    if common and identical:
        status = "SAME_ONTOLOGY_IRI_RDF_EQUIVALENT"
    elif common:
        status = "SAME_ONTOLOGY_IRI_DIFFERENT_RDF_GRAPHS_REVIEW_REQUIRED"
    else:
        status = "NO_SHARED_ONTOLOGY_IRI_REVIEW_IMPORT_MAPPING"
    report = {
        "status": status,
        "read_only": True,
        "files": records,
        "iao_comparison": {
            "left": first,
            "right": second,
            "shared_ontology_iris": common,
            "rdf_isomorphic": identical,
            "same_bytes": records[first]["sha256"] == records[second]["sha256"],
            "left_only_named_subclass_count": len({(str(s), str(o)) for s, o in graphs[first].subject_objects(RDFS.subClassOf) if isinstance(s, URIRef) and isinstance(o, URIRef)} - {(str(s), str(o)) for s, o in graphs[second].subject_objects(RDFS.subClassOf) if isinstance(s, URIRef) and isinstance(o, URIRef)}),
            "right_only_named_subclass_count": len({(str(s), str(o)) for s, o in graphs[second].subject_objects(RDFS.subClassOf) if isinstance(s, URIRef) and isinstance(o, URIRef)} - {(str(s), str(o)) for s, o in graphs[first].subject_objects(RDFS.subClassOf) if isinstance(s, URIRef) and isinstance(o, URIRef)}),
            "not_a_reasoner_validation": True,
        },
        "interpretation": (
            "Exact RDF equivalence allows a potential alias review; do not modify the audit automatically."
            if identical else
            "Two non-equivalent local RDF documents declare the same ontology IRI. Inspect provenance and differences before resolving the ambiguity."
        ),
    }
    destination = root / OUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[v9 identity] Status: {status}", flush=True)
    print(f"[v9 identity] Report: {OUT}", flush=True)
    print("[v9 identity] Existing ontology and frozen model files were not changed", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    try:
        diagnose(args.project_root.resolve())
    except Exception as ex:
        print(f"[v9 identity] FAILED: {ex}", flush=True)
        raise SystemExit(1) from ex


if __name__ == "__main__":
    main()
