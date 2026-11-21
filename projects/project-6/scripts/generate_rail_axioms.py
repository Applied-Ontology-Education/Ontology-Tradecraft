#!/usr/bin/env python3
"""Project 6 Step 3: deterministic, provisional OWL TBox candidate generation.

Usage (from project-6 root):
    python scripts/generate_rail_axioms.py

Inputs created by earlier steps:
    src/module.ttl
    imports/*.ttl, *.owl, ...
    generated/rail_pattern_proposal.json

Outputs:
    generated/rail_axioms_candidates.ttl
    reports/step3_axiom_review.json
    reports/step3_summary.json

No API calls, named individuals, OWL restrictions, merge, or source mutations.
Candidate TTL is NOT automatically approved and MUST NOT be merged into source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

VERSION = "project6-rail-axioms-v1"
RAIL_STATION = "railway station"
RAIL_ROLE = "rail passenger role"
XSD = "http://www.w3.org/2001/XMLSchema#"
BFO_ROLE = URIRef("http://purl.obolibrary.org/obo/BFO_0000023")
BFO_PROCESS = URIRef("http://purl.obolibrary.org/obo/BFO_0000015")
LOCAL_NAME = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
SUPPORTED_EXTENSIONS = {".ttl", ".owl", ".rdf", ".xml", ".nt", ".n3", ".jsonld", ".trig"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8", newline="\n")
    tmp.replace(path)


def atomic_json(path: Path, obj: dict) -> None:
    atomic_text(path, json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_graph(path: Path) -> Graph:
    choices = {
        ".ttl": ["turtle"], ".owl": ["xml", "turtle"],
        ".rdf": ["xml", "turtle"], ".xml": ["xml"],
        ".nt": ["nt"], ".n3": ["n3"],
        ".jsonld": ["json-ld"], ".trig": ["trig"],
    }.get(path.suffix.lower(), ["turtle", "xml"])
    problems = []
    for fmt in choices:
        try:
            g = Graph()
            g.parse(location=str(path), format=fmt)
            return g
        except Exception as e:
            problems.append(f"{fmt}: {e}")
    raise ValueError(f"Could not parse {path}: " + " | ".join(problems)[:500])


def iri(value: str, *, context: str) -> URIRef:
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValueError(f"{context} is not a valid absolute IRI: {value!r}")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc or any(c in value for c in '<>"{}|\\^`'):
        raise ValueError(f"{context} is not a valid HTTP(S) IRI: {value!r}")
    return URIRef(value)


def name(value: str, *, context: str) -> str:
    if not isinstance(value, str) or not LOCAL_NAME.fullmatch(value):
        raise ValueError(f"{context}: local_name must be lowercase snake_case, got {value!r}")
    return value


def text_field(obj: dict, field: str, *, context: str) -> str:
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: missing nonempty {field}")
    return value.strip()


def named_declarations(g: Graph, types: tuple) -> set[URIRef]:
    return {s for t in types for s in g.subjects(RDF.type, t) if isinstance(s, URIRef)}


def _detect_cycle(edges: dict[URIRef, URIRef]) -> None:
    for start in edges:
        visited = set()
        node = start
        while node in edges:
            if node in visited:
                raise ValueError(f"Cycle among candidate subclass edges, involving {node}")
            visited.add(node)
            node = edges[node]


def _emit_turtle(g: Graph) -> str:
    # N-Triples statements are also legal Turtle. No blank nodes are created;
    # sorted fully expanded terms guarantee byte-for-byte deterministic output.
    triples = sorted(f"{s.n3()} {p.n3()} {o.n3()} ." for s, p, o in g)
    return ("# Project 6 Step 3: UNAPPROVED rail-travel candidate TBox\n"
            "# DO NOT merge automatically into src/module.ttl. Review JSON is mandatory.\n"
            "# No instances or OWL restriction axioms are generated in this step.\n\n"
            + "\n".join(triples) + "\n")


def generate(root: Path) -> dict:
    root = root.resolve()
    source_path = root / "src/module.ttl"
    proposal_path = root / "generated/rail_pattern_proposal.json"
    if not source_path.is_file() or not proposal_path.is_file():
        raise FileNotFoundError("Expected src/module.ttl and generated/rail_pattern_proposal.json; run Steps 1 and 2 first")

    source_hash = sha256(source_path)
    source = parse_graph(source_path)
    document = load_json(proposal_path)
    if document.get("status") != "PROPOSED_NOT_VALIDATED_OWL":
        raise ValueError("Step 2 proposal status is not PROPOSED_NOT_VALIDATED_OWL")
    provenance = document.get("provenance", {})
    if provenance.get("source_sha256") != source_hash:
        raise ValueError("Proposal source SHA-256 differs from src/module.ttl; do not apply a stale proposal")
    namespace = document.get("namespace", "")
    iri(namespace + "probe", context="Proposal namespace")
    if not namespace.endswith(("#", "/")):
        raise ValueError("Proposed namespace must end in '#' or '/'")
    proposal = document.get("proposal")
    if not isinstance(proposal, dict):
        raise ValueError("Missing proposal object")

    import_hashes = provenance.get("local_import_hashes")
    if not isinstance(import_hashes, dict) or not import_hashes:
        raise ValueError("Missing import hash provenance from Step 2")
    import_dir = root / "imports"
    on_disk = sorted(p for p in import_dir.rglob("*")
                     if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    relative_paths = {"imports/" + p.relative_to(import_dir).as_posix(): p for p in on_disk}
    if set(relative_paths) != set(import_hashes):
        raise ValueError("Local import file set changed since Step 2: "
                         f"added={sorted(set(relative_paths)-set(import_hashes))}, "
                         f"missing={sorted(set(import_hashes)-set(relative_paths))}")

    declared_classes = named_declarations(source, (OWL.Class, RDFS.Class))
    existing_properties = named_declarations(source, (OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty))
    existing_individuals = named_declarations(source, (OWL.NamedIndividual,))
    occupied = {s for s in source.subjects() if isinstance(s, URIRef)}
    all_reference = Graph()
    for triple in source:
        all_reference.add(triple)
    for relative, path in sorted(relative_paths.items()):
        if sha256(path) != import_hashes[relative]:
            raise ValueError(f"Import changed since Step 2: {relative}; rerun Step 2")
        g = parse_graph(path)
        for triple in g:
            all_reference.add(triple)
        occupied.update(s for s in g.subjects() if isinstance(s, URIRef))
        declared_classes.update(named_declarations(g, (OWL.Class, RDFS.Class)))
        existing_properties.update(named_declarations(g, (OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty)))
        existing_individuals.update(named_declarations(g, (OWL.NamedIndividual,)))

    entries = proposal.get("rail_classes")
    obj_props = proposal.get("object_properties")
    data_props = proposal.get("data_properties")
    if not isinstance(entries, list) or not entries or not isinstance(obj_props, list) or not isinstance(data_props, list):
        raise ValueError("Proposal requires nonempty rail_classes and lists of object/data properties")

    rail_classes: dict[str, URIRef] = {}
    fresh_classes: dict[URIRef, dict] = {}
    edges: dict[URIRef, URIRef] = {}
    used_local_names = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("rail_classes must contain objects")
        target = text_field(item, "target_concept", context="rail_classes")
        if target in rail_classes:
            raise ValueError(f"Duplicate rail target: {target}")
        decision = item.get("decision")
        if decision == "CREATE":
            local = name(item.get("local_name"), context=target)
            if local in used_local_names:
                raise ValueError(f"Duplicate local name: {local}")
            used_local_names.add(local)
            entity = iri(namespace + local, context=target)
            if entity in occupied or entity in fresh_classes:
                raise ValueError(f"Proposed class IRI already used by loaded ontology: {entity}")
            fresh_classes[entity] = item
        elif decision == "REUSE":
            entity = iri(item.get("reused_iri"), context=target + " reused_iri")
            if entity not in declared_classes:
                raise ValueError(f"Reused class not declared in loaded ontology: {entity}")
        else:
            raise ValueError(f"Unsupported decision for {target}: {decision}")
        if entity in rail_classes.values():
            raise ValueError(f"Rail targets resolve to duplicate class IRI: {entity}")
        rail_classes[target] = entity

    allowed_classes = declared_classes | set(fresh_classes)
    for entity, item in fresh_classes.items():
        parent = iri(item.get("parent_iri"), context=f"parent for {entity}")
        if parent not in allowed_classes:
            raise ValueError(f"Superclass not declared in source/imports/candidates: {parent}")
        if entity == parent:
            raise ValueError(f"Class cannot be its own superclass: {entity}")
        edges[entity] = parent
    _detect_cycle(edges)

    required_parents = {"rail journey": BFO_PROCESS, "rail passenger role": BFO_ROLE}
    for target, expected in required_parents.items():
        entity = rail_classes.get(target)
        if entity is not None and entity in fresh_classes and edges[entity] != expected:
            raise ValueError(f"{target} must have direct superclass {expected}")

    candidates = Graph()
    audit_rows = []
    for target in sorted(rail_classes):
        entity = rail_classes[target]
        if entity not in fresh_classes:
            audit_rows.append({"kind": "class", "target": target, "iri": str(entity), "status": "REUSED_EXISTING"})
            continue
        item = fresh_classes[entity]
        label = text_field(item, "label", context=target)
        definition = text_field(item, "definition", context=target)
        parent = edges[entity]
        candidates.add((entity, RDF.type, OWL.Class))
        candidates.add((entity, RDFS.subClassOf, parent))
        candidates.add((entity, RDFS.label, Literal(label, lang="en")))
        candidates.add((entity, SKOS.definition, Literal(definition, lang="en")))
        notes = []
        if target == RAIL_STATION:
            notes.append("BLOCKING REVIEW: station is modeled as a building; some railway stations are larger facilities or sites")
            notes.append("Review the definition and superclass before accepting these axioms")
        elif target == RAIL_ROLE:
            notes.append("REVIEW: existing passenger role may be a more specific superclass than direct BFO Role")
        elif target == "rail ticket":
            notes.append("REVIEW: material bearer vs digital ticket information must be defined for scope")
        elif target == "train":
            notes.append("REVIEW: single BFO object vs multi-car vehicle aggregate")
        audit_rows.append({"kind": "class", "target": target, "iri": str(entity),
                           "parent_iri": str(parent),
                           "status": "REVIEW_REQUIRED" if notes else "CANDIDATE_UNAPPROVED",
                           "review_notes": notes})

    seen_new = set(fresh_classes)
    for kind, items, owl_type in (("object_property", obj_props, OWL.ObjectProperty),
                                  ("data_property", data_props, OWL.DatatypeProperty)):
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"{kind} entry must be an object")
            local = name(item.get("local_name"), context=kind)
            if local in used_local_names:
                raise ValueError(f"Repeated local name across generated entities: {local}")
            used_local_names.add(local)
            entity = iri(namespace + local, context=kind)
            if entity in occupied or entity in seen_new or entity in existing_properties or entity in existing_individuals:
                raise ValueError(f"Property IRI already exists or collides: {entity}")
            seen_new.add(entity)
            domain = iri(item.get("domain_iri"), context=local + " domain")
            if domain not in allowed_classes:
                raise ValueError(f"Unknown domain class for {local}: {domain}")
            range_value = item.get("range_iri" if kind == "object_property" else "datatype_iri")
            range_iri = iri(range_value, context=local + " range/datatype")
            if kind == "object_property" and range_iri not in allowed_classes:
                raise ValueError(f"Unknown range class for {local}: {range_iri}")
            if kind == "data_property" and not str(range_iri).startswith(XSD):
                raise ValueError(f"Unsupported datatype for {local}: {range_iri}; expected xsd:* URI")
            candidates.add((entity, RDF.type, owl_type))
            candidates.add((entity, RDFS.domain, domain))
            candidates.add((entity, RDFS.range, range_iri))
            candidates.add((entity, RDFS.label, Literal(text_field(item, "label", context=local), lang="en")))
            candidates.add((entity, SKOS.definition, Literal(text_field(item, "definition", context=local), lang="en")))
            notes = []
            if rail_classes.get(RAIL_STATION) in (domain, range_iri):
                notes.append("BLOCKING REVIEW: depends on contested railway_station class design")
            audit_rows.append({"kind": kind, "iri": str(entity), "local_name": local,
                               "domain_iri": str(domain), "range_iri": str(range_iri),
                               "status": "REVIEW_REQUIRED" if notes else "CANDIDATE_UNAPPROVED",
                               "review_notes": notes})

    if any(True for _ in candidates.subjects(RDF.type, OWL.NamedIndividual)) or any(isinstance(t, BNode) for triple in candidates for t in triple):
        raise ValueError("Candidate generator unexpectedly created an individual or blank node")
    content = _emit_turtle(candidates)
    parse_back = Graph()
    parse_back.parse(data=content, format="turtle")
    if set(parse_back) != set(candidates):
        raise RuntimeError("Turtle parse-back changed the candidate RDF graph")
    if sha256(source_path) != source_hash:
        raise RuntimeError("Source changed while generating candidates; refusing to write reports")

    reviewed = [r for r in audit_rows if r["status"] == "REVIEW_REQUIRED"]
    review = {
        "status": "REVIEW_REQUIRED_BEFORE_MERGE",
        "version": VERSION,
        "source_sha256": source_hash,
        "proposal_sha256": sha256(proposal_path),
        "rows": sorted(audit_rows, key=lambda x: (x["kind"], x["iri"])),
        "review_required_count": len(reviewed),
        "notes": [
            "A candidate is NOT an approved axiom; review every class and property before merging.",
            "No owl:equivalentClass or owl:Restriction axioms were proposed in Step 2, so none are fabricated here.",
            "RDFS domain/range restrictions can entail unintended classifications; review them and check reasoner consistency later.",
            "The railway_station Building parent and existing passenger-role alignment remain open modeling decisions.",
        ],
    }
    summary = {
        "status": "PASS_CANDIDATES_REVIEW_REQUIRED" if reviewed else "PASS_CANDIDATES_UNAPPROVED",
        "version": VERSION,
        "candidate_classes": len(fresh_classes),
        "reused_classes": len(rail_classes) - len(fresh_classes),
        "candidate_object_properties": len(obj_props),
        "candidate_data_properties": len(data_props),
        "candidate_triples": len(candidates),
        "review_required_items": len(reviewed),
        "source_sha256": source_hash,
        "source_unchanged": sha256(source_path) == source_hash,
        "no_named_individuals_created": True,
        "no_api_call": True,
        "merged_into_source": False,
    }
    atomic_text(root / "generated/rail_axioms_candidates.ttl", content)
    atomic_json(root / "reports/step3_axiom_review.json", review)
    atomic_json(root / "reports/step3_summary.json", summary)
    if sha256(source_path) != source_hash:
        raise RuntimeError("Source changed during report writing")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = generate(args.project_root)
    except Exception as exc:
        print(f"STEP 3 — FAIL: {exc}", file=sys.stderr)
        return 1
    print("STEP 3 — " + result["status"])
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
