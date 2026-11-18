#!/usr/bin/env python3
"""Project 6 Step 3A: deterministic, auditable refinement of Step 3 rail OWL TBox.

Run from project-6 root:
    python scripts/refine_rail_axioms.py

Requires upstream Steps 1-3. Does not call an API or modify any input.
Outputs:
    generated/rail_axioms_refined.ttl  (UNAPPROVED; not merged)
    reports/step3a_refinement_audit.json
    reports/step3a_summary.json

The refinements implemented here are *provisional modeling decisions*, not
inferences, logical validation, or blanket approval for merging. Do not treat
these outputs as reasoner-validated until later QC/reasoner stages succeed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

# Reuse the Step 3 file parser and atomic writers instead of inventing a second
# import parser; Step 3 must already be installed for the pipeline to run.
from generate_rail_axioms import (
    SUPPORTED_EXTENSIONS, atomic_json, atomic_text, parse_graph, sha256,
)

VERSION = "project6-rail-refinement-v1"
BASE_NS = "https://example.org/ontologies/intel-ontology#"
AGENT_NS = "https://www.commoncoreontologies.org/AgentOntology/"
BFO_MATERIAL = URIRef("http://purl.obolibrary.org/obo/BFO_0000040")
BFO_OBJECT = URIRef("http://purl.obolibrary.org/obo/BFO_0000030")
BFO_ROLE = URIRef("http://purl.obolibrary.org/obo/BFO_0000023")
EXISTING_PASSENGER = URIRef(AGENT_NS + "passenger")
EXISTING_BUILDING = URIRef(AGENT_NS + "building")

TRAIN_DEF = (
    "A material entity composed of one or more rail vehicles operated together "
    "on a railway to convey passengers or freight."
)
STATION_DEF = (
    "A material entity comprising physical railway facilities at which rail "
    "journeys begin, end, or pause, potentially including platforms, track "
    "infrastructure, structures, and buildings."
)


def _required_files(root: Path) -> dict[str, Path]:
    files = {
        "source": root / "src/module.ttl",
        "proposal": root / "generated/rail_pattern_proposal.json",
        "candidate": root / "generated/rail_axioms_candidates.ttl",
        "step3_review": root / "reports/step3_axiom_review.json",
        "step3_summary": root / "reports/step3_summary.json",
    }
    missing = [str(p.relative_to(root)) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing Step 3A prerequisite(s): " + ", ".join(missing))
    return files


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _verify_inputs(root: Path, files: dict[str, Path]) -> tuple[Graph, dict, dict, str]:
    source_hash = sha256(files["source"])
    proposal_hash = sha256(files["proposal"])
    source = parse_graph(files["source"])
    proposal = _load_json(files["proposal"])
    review = _load_json(files["step3_review"])
    summary = _load_json(files["step3_summary"])
    provenance = proposal.get("provenance", {})
    if (proposal.get("status") != "PROPOSED_NOT_VALIDATED_OWL" or
            proposal.get("namespace") != BASE_NS):
        raise ValueError("Unexpected Step 2 proposal status/namespace; review inputs")
    if not (source_hash == provenance.get("source_sha256") == review.get("source_sha256")
            == summary.get("source_sha256")):
        raise ValueError("Source hash mismatch across Steps 2/3: rerun upstream steps")
    if review.get("proposal_sha256") != proposal_hash:
        raise ValueError("Step 3 review references a different Step 2 proposal")
    if (summary.get("status") != "PASS_CANDIDATES_REVIEW_REQUIRED" or
            summary.get("merged_into_source") is not False or
            summary.get("no_named_individuals_created") is not True):
        raise ValueError("Unexpected Step 3 summary; refusing refinement")
    if review.get("status") != "REVIEW_REQUIRED_BEFORE_MERGE":
        raise ValueError("Unexpected Step 3 review status")

    hashes = provenance.get("local_import_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("No local-import hashes in Step 2 provenance")
    imports = root / "imports"
    found = {
        "imports/" + p.relative_to(imports).as_posix(): p
        for p in imports.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    } if imports.is_dir() else {}
    if set(found) != set(hashes):
        raise ValueError("Import file set changed since Step 2; rerun upstream steps")
    imported = Graph()
    for rel, file in sorted(found.items()):
        if sha256(file) != hashes[rel]:
            raise ValueError(f"Import hash mismatch for {rel}; rerun upstream steps")
        for triple in parse_graph(file):
            imported.add(triple)

    # The BFO material superclass must really exist in the imported ontology,
    # rather than merely being an IRI typed into the refinement script.
    if ((BFO_MATERIAL, RDF.type, OWL.Class) not in imported and
            (BFO_MATERIAL, RDF.type, RDFS.Class) not in imported):
        raise ValueError("BFO Material Entity not declared in local imports")
    # Verify the exact existing role pattern in the user's source ontology.
    if ((EXISTING_PASSENGER, RDF.type, OWL.Class) not in source or
            (EXISTING_PASSENGER, RDFS.subClassOf, BFO_ROLE) not in source):
        raise ValueError("Existing passenger class must be a declared subclass of BFO Role")
    return source, proposal, summary, source_hash



def _verify_candidate_matches_proposal(graph: Graph, document: dict) -> None:
    """Reject same-count hand edits: recompute Step 3's *exact* RDF triples.

    This is deliberately limited to the Step 2 CREATE-only design the user
    reviewed. Extending the schema will require an explicit policy update.
    """
    part = document.get("proposal", {})
    expected = Graph()
    classes = part.get("rail_classes", [])
    obj_props = part.get("object_properties", [])
    data_props = part.get("data_properties", [])
    if (not isinstance(classes, list) or len(classes) != 8 or
            not isinstance(obj_props, list) or len(obj_props) != 4 or
            not isinstance(data_props, list) or len(data_props) != 2):
        raise ValueError("Step 3A expects the eight-class, six-property proposal")
    for item in classes:
        if item.get("decision") != "CREATE":
            raise ValueError("Step 3A policy assumes all eight rail classes are CREATE")
        subject = URIRef(BASE_NS + item["local_name"])
        expected.add((subject, RDF.type, OWL.Class))
        expected.add((subject, RDFS.subClassOf, URIRef(item["parent_iri"])))
        expected.add((subject, RDFS.label, Literal(item["label"].strip(), lang="en")))
        expected.add((subject, SKOS.definition, Literal(item["definition"].strip(), lang="en")))
    for members, typ, range_key in (
        (obj_props, OWL.ObjectProperty, "range_iri"),
        (data_props, OWL.DatatypeProperty, "datatype_iri"),
    ):
        for item in members:
            subject = URIRef(BASE_NS + item["local_name"])
            expected.add((subject, RDF.type, typ))
            expected.add((subject, RDFS.domain, URIRef(item["domain_iri"])))
            expected.add((subject, RDFS.range, URIRef(item[range_key])))
            expected.add((subject, RDFS.label, Literal(item["label"].strip(), lang="en")))
            expected.add((subject, SKOS.definition, Literal(item["definition"].strip(), lang="en")))
    if set(graph) != set(expected):
        raise ValueError(
            "Step 3 candidate content differs from the Step 2 proposal; "
            "rerun Step 3 instead of editing generated axioms"
        )

def _single_object(graph: Graph, subject: URIRef, predicate: URIRef, expected) -> None:
    found = set(graph.objects(subject, predicate))
    if found != {expected}:
        raise ValueError(
            f"Unexpected Step 3 candidate axiom for {subject} {predicate}: "
            f"expected exactly {expected!r}; found {sorted(map(str, found))}"
        )


def _assert_no_individuals_or_blank_nodes(graph: Graph) -> None:
    if any(True for _ in graph.subjects(RDF.type, OWL.NamedIndividual)):
        raise ValueError("Refined candidate contains named individuals")
    if any(isinstance(term, BNode) for triple in graph for term in triple):
        raise ValueError("Refined candidate contains anonymous individuals/restrictions")


def _replace_object(graph: Graph, subject: URIRef, predicate: URIRef,
                    old, new, reason: str, changes: list[dict]) -> None:
    _single_object(graph, subject, predicate, old)
    graph.remove((subject, predicate, old))
    graph.add((subject, predicate, new))
    changes.append({
        "action": "REPLACE_TRIPLE_OBJECT", "subject": str(subject),
        "predicate": str(predicate), "before": str(old), "after": str(new),
        "reason": reason,
    })


def _emit(graph: Graph) -> str:
    lines = sorted(f"{s.n3()} {p.n3()} {o.n3()} ." for s, p, o in graph)
    return (
        "# Project 6 Step 3A — REFINED BUT UNAPPROVED TBox candidate axioms\n"
        "# No source merge, no named individuals, no reasoner approval.\n"
        "# See reports/step3a_refinement_audit.json for every changed triple.\n\n"
        + "\n".join(lines) + "\n"
    )


def refine(root: Path) -> dict:
    root = root.resolve()
    files = _required_files(root)
    originals = {k: sha256(v) for k, v in files.items()}
    source, proposal, summary3, source_hash = _verify_inputs(root, files)

    baseline = parse_graph(files["candidate"])
    _assert_no_individuals_or_blank_nodes(baseline)
    _verify_candidate_matches_proposal(baseline, proposal)
    if len(baseline) != summary3.get("candidate_triples"):
        raise ValueError("Step 3 candidate count differs from its summary")
    if len(baseline) != 62:
        raise ValueError("Expected 62 original candidate triples for this Step 3A policy")

    def entity(local: str) -> URIRef:
        return URIRef(BASE_NS + local)

    train = entity("train")
    station = entity("railway_station")
    role = entity("rail_passenger_role")
    ticket = entity("rail_ticket")
    for e in (train, station, role, ticket):
        _single_object(baseline, e, RDF.type, OWL.Class)
    _single_object(baseline, train, RDFS.subClassOf, BFO_OBJECT)
    _single_object(baseline, station, RDFS.subClassOf, EXISTING_BUILDING)
    _single_object(baseline, role, RDFS.subClassOf, BFO_ROLE)
    _single_object(baseline, ticket, RDFS.subClassOf, URIRef(AGENT_NS + "ticket"))

    # No assumptions about the wording generated by the model: derive the
    # existing definition from the validated candidate graph itself.
    def old_definition(subject: URIRef) -> Literal:
        values = set(baseline.objects(subject, SKOS.definition))
        if len(values) != 1 or not all(isinstance(v, Literal) and v.language == "en" for v in values):
            raise ValueError(f"Expected exactly one English SKOS definition for {subject}")
        return next(iter(values))

    # These edits are fixed, intentional refinement policies. The unmodified
    # Step 2 proposal and Step 3 candidate file remain the provenance record.
    refined = Graph()
    for triple in baseline:
        refined.add(triple)
    edits: list[dict] = []
    _replace_object(refined, train, RDFS.subClassOf, BFO_OBJECT, BFO_MATERIAL,
                    "Accommodate trains assembled from multiple rail vehicles", edits)
    _replace_object(refined, station, RDFS.subClassOf, EXISTING_BUILDING, BFO_MATERIAL,
                    "A station facility can include platforms and infrastructure, not only a building", edits)
    _replace_object(refined, train, SKOS.definition, old_definition(train),
                    Literal(TRAIN_DEF, lang="en"),
                    "Describe single- and multi-vehicle trains using material entity", edits)
    _replace_object(refined, station, SKOS.definition, old_definition(station),
                    Literal(STATION_DEF, lang="en"),
                    "Broaden definition to cover railway facilities", edits)
    if (role, RDFS.subClassOf, EXISTING_PASSENGER) in refined:
        raise ValueError("Passenger-role alignment is already present; baseline changed")
    refined.add((role, RDFS.subClassOf, EXISTING_PASSENGER))
    edits.append({
        "action": "ADD_TRIPLE", "subject": str(role),
        "predicate": str(RDFS.subClassOf), "before": None,
        "after": str(EXISTING_PASSENGER),
        "reason": "Reuse existing passenger Role subclass while retaining explicit BFO Role parent",
    })

    _assert_no_individuals_or_blank_nodes(refined)
    if set(refined) - set(baseline) != {
        (train, RDFS.subClassOf, BFO_MATERIAL),
        (station, RDFS.subClassOf, BFO_MATERIAL),
        (role, RDFS.subClassOf, EXISTING_PASSENGER),
        (train, SKOS.definition, Literal(TRAIN_DEF, lang="en")),
        (station, SKOS.definition, Literal(STATION_DEF, lang="en")),
    }:
        raise RuntimeError("Refinement added unexpected triples")
    if len(refined) != len(baseline) + 1:
        raise RuntimeError("Expected one net new triple for role alignment")

    output = _emit(refined)
    parsed = Graph().parse(data=output, format="turtle")
    if set(parsed) != set(refined):
        raise RuntimeError("Refined Turtle round-trip changed the RDF graph")
    for key, path in files.items():
        if sha256(path) != originals[key]:
            raise RuntimeError(f"Input changed during Step 3A: {key}; abort")

    report = {
        "status": "REFINED_UNAPPROVED",
        "version": VERSION,
        "source_sha256": source_hash,
        "source_triples": len(source),
        "proposal_sha256": originals["proposal"],
        "step3_candidate_sha256": originals["candidate"],
        "step3_review_sha256": originals["step3_review"],
        "baseline_triples": len(baseline),
        "refined_triples": len(refined),
        "edits": edits,
        "resolved_step3_flags": [
            "train: material-entity parent and revised definition",
            "railway station: broader material-entity parent and revised definition",
            "rail passenger role: existing passenger superclass added",
            "three station-related object properties: existing domains/ranges retained after broader station class",
        ],
        "open_design_decisions": [
            {
                "entity": str(ticket),
                "decision": "Material information bearer scope retained; electronic ticket INFORMATION requires a distinct class/pattern",
                "blocking_for_unreviewed_merge": True,
            },
            {
                "entity": str(station),
                "decision": "An optional BFO Site / facility-vs-infrastructure distinction may be needed after formal review",
                "blocking_for_unreviewed_merge": False,
            },
        ],
        "validation_scope": "RDF graph/identity/traceability checks only; NOT OWL consistency or SHACL/SPARQL approval",
        "no_merge": True,
    }
    result = {
        "status": "PASS_REFINED_UNAPPROVED",
        "version": VERSION,
        "baseline_triples": len(baseline),
        "refined_triples": len(refined),
        "refinement_operations": len(edits),
        "classes": len(list(refined.subjects(RDF.type, OWL.Class))),
        "object_properties": len(list(refined.subjects(RDF.type, OWL.ObjectProperty))),
        "data_properties": len(list(refined.subjects(RDF.type, OWL.DatatypeProperty))),
        "pending_scope_decisions": 2,
        "no_api_call": True,
        "no_named_individuals_created": True,
        "merged_into_source": False,
        "source_sha256": source_hash,
        "source_unchanged": sha256(files["source"]) == source_hash,
        "original_step3_candidate_unchanged": sha256(files["candidate"]) == originals["candidate"],
        "original_step2_proposal_unchanged": sha256(files["proposal"]) == originals["proposal"],
    }
    atomic_text(root / "generated/rail_axioms_refined.ttl", output)
    atomic_json(root / "reports/step3a_refinement_audit.json", report)
    atomic_json(root / "reports/step3a_summary.json", result)
    for key, path in files.items():
        if sha256(path) != originals[key]:
            raise RuntimeError(f"Input changed during output writing: {key}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = refine(args.project_root)
    except Exception as exc:
        print(f"STEP 3A — FAIL: {exc}", file=sys.stderr)
        return 1
    print("STEP 3A — " + result["status"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
