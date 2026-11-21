#!/usr/bin/env python3
"""Project 6, Step 1: read-only source/import inventory and rail-travel gap report.

Run from any working directory:
    python scripts/inventory_ontology.py

Only src/module.ttl is a required project data file. The imports/ folder is
inspected locally: no network requests, ontology merges, or axiom changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from rdflib import BNode, Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

SCRIPT_VERSION = "project6-rail-inventory-v1"
ONTOLOGY_EXTENSIONS = {".ttl", ".owl", ".rdf", ".xml", ".nt", ".n3", ".jsonld", ".trig"}
ENTITY_TYPES = {
    "classes": (OWL.Class, RDFS.Class),
    "object_properties": (OWL.ObjectProperty,),
    "data_properties": (OWL.DatatypeProperty,),
    "annotation_properties": (OWL.AnnotationProperty,),
    "individuals": (OWL.NamedIndividual,),
}

# These are pattern targets, not assertions about ontology content.
# Only exact normalized labels/local names count as a detected entity.
RAIL_PATTERN_TARGETS = [
    {
        "concept": "train",
        "aliases": ["train", "railway train"],
        "air_analogs": ["Airplane"],
        "design_note": "Material vehicle; check applicable BFO/CCO superclass.",
    },
    {
        "concept": "passenger train",
        "aliases": ["passenger train"],
        "air_analogs": ["Commcercial_airplane"],
        "design_note": "Specialization of train; confirm intended definition.",
    },
    {
        "concept": "railway station",
        "aliases": ["railway station", "train station", "rail station"],
        "air_analogs": ["airport"],
        "design_note": "Model facility and its location separately if necessary.",
    },
    {
        "concept": "rail journey",
        "aliases": ["rail journey", "train journey", "rail travel process"],
        "air_analogs": [],
        "design_note": "Actual travel process, distinct from journey plan information.",
    },
    {
        "concept": "rail journey plan",
        "aliases": ["rail journey plan", "train journey plan", "rail itinerary"],
        "air_analogs": ["Flight_plan"],
        "design_note": "Information content entity, not the travel process.",
    },
    {
        "concept": "rail ticket",
        "aliases": ["rail ticket", "train ticket"],
        "air_analogs": ["boarding_pass", "ticket"],
        "design_note": "Distinguish ticket information from a physical information bearer.",
    },
    {
        "concept": "rail reservation information",
        "aliases": ["rail reservation information", "train reservation information"],
        "air_analogs": ["airline_reservation", "reservation"],
        "design_note": "Reservation information ICE; not a physical booking artifact.",
    },
    {
        "concept": "rail passenger role",
        "aliases": ["rail passenger role", "train passenger role", "rail passenger"],
        "air_analogs": ["passenger", "Jetblue_passenger"],
        "design_note": "Role borne by a person, not a subtype of person.",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_graph(path: Path) -> Graph:
    """Parse ontology RDF without fetching any owl:imports links."""
    extension = path.suffix.lower()
    preferred = {
        ".ttl": ["turtle"],
        ".owl": ["xml", "turtle"],
        ".rdf": ["xml", "turtle"],
        ".xml": ["xml"],
        ".nt": ["nt"],
        ".n3": ["n3", "turtle"],
        ".jsonld": ["json-ld"],
        ".trig": ["trig"],
    }.get(extension, ["turtle", "xml"])
    errors = []
    for rdf_format in preferred:
        try:
            graph = Graph()
            graph.parse(location=str(path), format=rdf_format)
            return graph
        except Exception as exc:
            errors.append(f"{rdf_format}: {type(exc).__name__}: {str(exc)[:250]}")
    raise ValueError(f"Could not parse {path}: " + " | ".join(errors))


def local_name(iri: str) -> str:
    return iri.rsplit("#", 1)[-1].rstrip("/").rsplit("/", 1)[-1]


def canonical_term(s: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def label_values(graph: Graph, term: URIRef) -> list[str]:
    labels = {str(x) for pred in (RDFS.label, SKOS.prefLabel) for x in graph.objects(term, pred)}
    return sorted(labels)


def declarations(graph: Graph) -> dict[str, set[URIRef]]:
    found = {}
    for category, declared_types in ENTITY_TYPES.items():
        found[category] = {
            node for declared_type in declared_types
            for node in graph.subjects(RDF.type, declared_type)
            if isinstance(node, URIRef)
        }
    return found


def describe_entity(graph: Graph, iri: URIRef) -> dict:
    return {
        "iri": str(iri),
        "labels": label_values(graph, iri),
        "superclasses_asserted": sorted(str(o) for o in graph.objects(iri, RDFS.subClassOf) if isinstance(o, URIRef)),
        "domains_asserted": sorted(str(o) for o in graph.objects(iri, RDFS.domain) if isinstance(o, URIRef)),
        "ranges_asserted": sorted(str(o) for o in graph.objects(iri, RDFS.range) if isinstance(o, URIRef)),
    }


def matches_catalog(catalog: list[dict], aliases: list[str]) -> list[dict]:
    needle = {canonical_term(alias) for alias in aliases}
    hits = []
    for item in catalog:
        names = [local_name(item["iri"]), *item["labels"]]
        if any(canonical_term(name) in needle for name in names):
            hits.append({"iri": item["iri"], "labels": item["labels"], "origin": item["origin"]})
    return sorted(hits, key=lambda x: (x["origin"], x["iri"]))


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def audit_project(root: Path) -> dict:
    source = root / "src" / "module.ttl"
    imports_dir = root / "imports"
    reports_dir = root / "reports"
    if not source.is_file():
        raise FileNotFoundError(f"Missing required source ontology: {source}")
    if not imports_dir.is_dir():
        raise FileNotFoundError(f"Missing imports folder: {imports_dir}")

    source_digest_before = sha256_file(source)
    source_graph = parse_graph(source)
    source_decls = declarations(source_graph)
    expected_imports = sorted({str(o) for o in source_graph.objects(None, OWL.imports) if isinstance(o, URIRef)})
    source_inventory = {
        category: [describe_entity(source_graph, term) for term in sorted(terms, key=str)]
        for category, terms in source_decls.items()
    }
    source_catalog = [
        {**desc, "origin": "source"}
        for category in ("classes", "object_properties", "data_properties")
        for desc in source_inventory[category]
    ]

    import_paths = sorted(
        p for p in imports_dir.rglob("*") if p.is_file() and p.suffix.lower() in ONTOLOGY_EXTENSIONS
    )
    local_import_catalog: list[dict] = []
    import_summaries = []
    for path in import_paths:
        graph = parse_graph(path)  # Malformed local ontologies are a blocking error.
        decls = declarations(graph)
        local_import_catalog.extend(
            {**describe_entity(graph, term), "origin": f"imports/{path.relative_to(imports_dir).as_posix()}"}
            for category in ("classes", "object_properties", "data_properties")
            for term in sorted(decls[category], key=str)
        )
        import_summaries.append({
            "file": f"imports/{path.relative_to(imports_dir).as_posix()}",
            "sha256": sha256_file(path),
            "triples": len(graph),
            "declared_classes": len(decls["classes"]),
            "declared_object_properties": len(decls["object_properties"]),
            "declared_data_properties": len(decls["data_properties"]),
            "declared_ontology_iris": sorted(str(s) for s in graph.subjects(RDF.type, OWL.Ontology) if isinstance(s, URIRef)),
            "version_iris": sorted(str(o) for o in graph.objects(None, OWL.versionIRI) if isinstance(o, URIRef)),
        })
        del graph

    rail_analysis = []
    for target in RAIL_PATTERN_TARGETS:
        local_matches = matches_catalog(source_catalog, target["aliases"])
        imported_matches = matches_catalog(local_import_catalog, target["aliases"])
        analogs = []
        for name in target["air_analogs"]:
            analogs.extend(matches_catalog(source_catalog, [name]))
        rail_analysis.append({
            "concept": target["concept"],
            "status": "EXISTS_IN_SOURCE" if local_matches else (
                "FOUND_IN_LOCAL_IMPORTS" if imported_matches else "NOT_FOUND_IN_LOADED_FILES"
            ),
            "source_matches": local_matches,
            "import_matches": imported_matches,
            "air_pattern_analogs": sorted({x["iri"]: x for x in analogs}.values(), key=lambda x: x["iri"]),
            "design_note": target["design_note"],
        })

    gca_subclasses = sum(1 for s in source_graph.subjects(RDFS.subClassOf) if isinstance(s, BNode))
    gca_equivalents = sum(1 for s in source_graph.subjects(OWL.equivalentClass) if isinstance(s, BNode))
    inventory = {
        "script_version": SCRIPT_VERSION,
        "source_file": "src/module.ttl",
        "source_sha256": source_digest_before,
        "source_triples": len(source_graph),
        "declared_import_iris": expected_imports,
        "source_counts": {**{key: len(items) for key, items in source_inventory.items()},
                          "anonymous_subclass_gcas": gca_subclasses,
                          "anonymous_equivalence_gcas": gca_equivalents},
        "source_entities": source_inventory,
        "local_import_files": import_summaries,
        "notes": [
            "Imported files are inspected locally only; remote owl:imports are never fetched.",
            "A local imported file is not automatically proven equivalent to a declared owl:imports IRI.",
            "No OWL reasoner or logical consistency checking is performed in Step 1.",
            "Counts cover explicit declarations in source; they do not include inferred entities.",
        ],
    }
    gap_report = {
        "script_version": SCRIPT_VERSION,
        "purpose": "Lexical rail-domain inventory and gap report; not generated OWL axioms",
        "rail_concepts": rail_analysis,
        "rail_concepts_not_found": sum(x["status"] == "NOT_FOUND_IN_LOADED_FILES" for x in rail_analysis),
        "notes": [
            "EXISTS_IN_SOURCE or FOUND_IN_LOCAL_IMPORTS means an exact normalized name matched a declared class/property.",
            "NOT_FOUND_IN_LOADED_FILES does not prove a concept is absent from all external imports.",
            "The tool does not create new classes, individuals, properties, or axioms.",
        ],
    }
    if sha256_file(source) != source_digest_before:
        raise RuntimeError("Source file changed during the audit; aborting")

    summary = {
        "status": "PASS",
        "source_unchanged": True,
        "source_sha256": source_digest_before,
        "source_triples": len(source_graph),
        "named_classes": len(source_decls["classes"]),
        "named_individuals": len(source_decls["individuals"]),
        "anonymous_subclass_gcas": gca_subclasses,
        "object_properties": len(source_decls["object_properties"]),
        "data_properties": len(source_decls["data_properties"]),
        "declared_imports": len(expected_imports),
        "local_import_files_read": len(import_paths),
        "rail_concepts_not_found": gap_report["rail_concepts_not_found"],
        "warning": "No local import files supplied; external ontology terms have not been inventoried" if not import_paths else None,
    }
    atomic_json(reports_dir / "ontology_inventory.json", inventory)
    atomic_json(reports_dir / "rail_gap_analysis.json", gap_report)
    atomic_json(reports_dir / "step1_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        summary = audit_project(args.project_root.resolve())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"STEP 1 — FAIL: {exc}", file=sys.stderr)
        return 1
    print("STEP 1 — SOURCE INVENTORY AND RAIL GAP AUDIT")
    for k in ("status", "source_triples", "named_classes", "named_individuals", "anonymous_subclass_gcas", "object_properties", "data_properties", "declared_imports", "local_import_files_read", "rail_concepts_not_found", "source_unchanged"):
        print(f"{k:27s}: {summary[k]}")
    if summary["warning"]:
        print("WARNING:", summary["warning"])
    print("Reports written to reports/; source ontology untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
