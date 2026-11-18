#!/usr/bin/env python3
"""Project 6 Step 2: LLM-assisted RAIL TBox design proposal (NO OWL changes).

Run from project root:
    python scripts/propose_rail_patterns_llm.py --dry-run
    python scripts/propose_rail_patterns_llm.py

Step 1 is rerun automatically to guarantee fresh reports. This stage does not
create individuals, write OWL axioms, merge graphs, or modify src/module.ttl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from rdflib import URIRef
from rdflib.namespace import OWL, RDF, RDFS

from inventory_ontology import audit_project, declarations, parse_graph, sha256_file, atomic_json

VERSION = "project6-rail-proposal-v1"
DEFAULT_NAMESPACE = "https://example.org/ontologies/intel-ontology#"
BASE_TYPES = {
    "object": "http://purl.obolibrary.org/obo/BFO_0000030",
    "process": "http://purl.obolibrary.org/obo/BFO_0000015",
    "role": "http://purl.obolibrary.org/obo/BFO_0000023",
    "information_content_entity": "http://purl.obolibrary.org/obo/IAO_0000030",
    "material_information_bearer": "http://purl.obolibrary.org/obo/IAO_0000178",
}


def object_schema(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def string() -> dict:
    return {"type": "string"}


def strings() -> dict:
    return {"type": "array", "items": string()}


SCHEMA = object_schema({
    "rail_classes": {"type": "array", "items": object_schema({
        "target_concept": string(),
        "decision": {"type": "string", "enum": ["CREATE", "REUSE"]},
        "local_name": string(),
        "label": string(),
        "definition": string(),
        "parent_iri": string(),
        "reused_iri": string(),
        "analogue_iris": strings(),
        "rationale": string(),
        "review_notes": strings(),
    })},
    "object_properties": {"type": "array", "items": object_schema({
        "local_name": string(),
        "label": string(),
        "definition": string(),
        "domain_iri": string(),
        "range_iri": string(),
        "reused_analogue_iri": string(),
        "rationale": string(),
    })},
    "data_properties": {"type": "array", "items": object_schema({
        "local_name": string(),
        "label": string(),
        "definition": string(),
        "domain_iri": string(),
        "datatype_iri": string(),
        "rationale": string(),
    })},
    "design_warnings": strings(),
})

SYSTEM_PROMPT = """You are advising on a conservative BFO/IAO/CCO-conformant OWL TBox design.
Return JSON satisfying the provided schema. This is a candidate design report, NOT approved ontology assertions.
Never invent IRIs for an existing BFO/IAO/CCO entity; use only the supplied reference vocabulary.
Create one rail_classes item for EACH target concept supplied (exact target_concept spelling).
For concepts with exact-name existing matches, choose REUSE and select a supplied match IRI.
For concepts with no matches, choose CREATE: use a unique lowercase snake_case local_name;
set reused_iri to an empty string, and select a known class IRI as parent_iri.
For REUSE set local_name, parent_iri, definition to empty strings; reused_iri must match supplied options.
A proposed class IRI is namespace + local_name. You may use proposed class IRIs as the
parent of a proposed child class, or as domain/range of new proposed object properties.
Use the offered air analogues as comparisons only, not owl:equivalentClass assertions.
Distinguish actual train journey (BFO process), journey plan/reservation information
(information content entities), rail ticket as material information bearer (if that is
what is intended), train/station physical objects, and passenger role (BFO role,
NOT person). Do not create any named individuals, social-media threat classifications,
trip records, or statements about specific persons. Keep BFO role and person distinct.
MANDATORY: For the target_concept 'rail passenger role', use parent_iri exactly
'http://purl.obolibrary.org/obo/BFO_0000023' (BFO Role), NOT passenger.
MANDATORY: For the target_concept 'rail journey', use parent_iri exactly
'http://purl.obolibrary.org/obo/BFO_0000015' (BFO Process).
Propose 2-4 NEW object properties relevant to rail processes/facilities/vehicle and
0-2 relevant new datatype properties; avoid duplicating existing properties.
Property domains/ranges must be known class IRIs or newly proposed class IRIs.
Never use invented external namespace IRIs or existing named individuals as class parents.
Express uncertainties and potential modeling pitfalls in review_notes/design_warnings.
Do not output Turtle, RDF/XML, SPARQL, or any generated OWL axioms in this stage."""


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def catalog_and_imports(root: Path) -> tuple[set[str], set[str], set[str], dict[str, dict]]:
    """Return known class/property/individual IRIs and local ontology identifiers."""
    classes, properties, individuals = set(), set(), set()
    identifiers = {}
    paths = [root / "src/module.ttl"]
    for path in sorted((root / "imports").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".ttl", ".owl", ".rdf", ".xml", ".nt", ".n3", ".trig", ".jsonld"}:
            paths.append(path)
    for path in paths:
        graph = parse_graph(path)
        decl = declarations(graph)
        classes.update(str(t) for t in decl["classes"])
        properties.update(str(t) for k in ("object_properties", "data_properties", "annotation_properties") for t in decl[k])
        individuals.update(str(t) for t in decl["individuals"])
        if path != paths[0]:
            ids = {str(o) for o in graph.subjects(RDF.type, OWL.Ontology) if isinstance(o, URIRef)}
            ids.update(str(o) for o in graph.objects(None, OWL.versionIRI) if isinstance(o, URIRef))
            identifiers[path.relative_to(root).as_posix()] = {"identities": sorted(ids), "sha256": sha256_file(path)}
    return classes, properties, individuals, identifiers


def build_request(root: Path, namespace: str, model: str) -> tuple[dict, set[str], set[str], set[str]]:
    # Rerunning inventory protects against stale JSON reports after import edits.
    summary = audit_project(root)
    inventory = load_json(root / "reports/ontology_inventory.json")
    gap = load_json(root / "reports/rail_gap_analysis.json")
    classes, properties, individuals, imports = catalog_and_imports(root)
    declared = set(inventory["declared_import_iris"])
    resolved = {iri for meta in imports.values() for iri in meta["identities"]}
    unresolved = sorted(declared - resolved)
    if unresolved:
        raise ValueError("Declared imports without matching local ontology/version IRI: " + ", ".join(unresolved))
    if not namespace.startswith("https://") or not namespace.endswith(("#", "/")):
        raise ValueError("--namespace must be an HTTPS IRI ending in # or /")
    if set(BASE_TYPES.values()) - classes:
        raise ValueError("Missing BFO/IAO reference classes: " + ", ".join(sorted(set(BASE_TYPES.values()) - classes)))

    source_classes = inventory["source_entities"]["classes"]
    source_props = inventory["source_entities"]["object_properties"]
    source_data = inventory["source_entities"]["data_properties"]
    # Exact IRIs rather than invented labels for imported vocabulary.
    base_labels = {v: k for k, v in BASE_TYPES.items()}
    parent_choices = [{"iri": iri, "label": base_labels[iri]} for iri in sorted(base_labels)]
    optional_relevant = [iri for iri in sorted(classes) if
                         any(k in iri.lower() for k in ("vehicle", "station", "journey", "transport", "ticket", "rail"))]
    parent_choices += [{"iri": iri, "label": "additional known imported class"}
                       for iri in optional_relevant[:80] if iri not in base_labels]
    payload = {
        "goal": "Project 6 Option 1: propose and align rail-travel TBox patterns with existing air-travel ontology",
        "namespace": namespace,
        "source_hash": summary["source_sha256"],
        "input_source_triples": summary["source_triples"],
        "local_imports": imports,
        "model": model,
        "pattern_targets": gap["rail_concepts"],
        "available_parent_classes": parent_choices,
        "source_classes": source_classes,
        "source_object_properties": source_props,
        "source_data_properties": source_data,
        "constraint": "TBox proposals only, preserve source, no new individuals",
    }
    return payload, classes, properties, individuals


def validate_proposal(response: dict, payload: dict, classes: set[str],
                      properties: set[str], individuals: set[str]) -> dict:
    """Fail closed on schema violations, ungrounded parent IRIs, and symbol collisions."""
    if not isinstance(response, dict) or set(response) != set(SCHEMA["properties"]):
        raise ValueError("Response has missing/unexpected top-level keys")
    for key in ("rail_classes", "object_properties", "data_properties", "design_warnings"):
        if not isinstance(response[key], list):
            raise ValueError(f"{key} must be a list")
    namespace = payload["namespace"]
    targets = {x["concept"]: x for x in payload["pattern_targets"]}
    decisions = response["rail_classes"]
    if len(decisions) != len(targets):
        raise ValueError(f"Expected exactly {len(targets)} rail_class decisions")
    seen_targets, proposed_iris = set(), set()
    for item in decisions:
        if not isinstance(item, dict) or set(item) != set(SCHEMA["properties"]["rail_classes"]["items"]["properties"]):
            raise ValueError("rail_classes item has missing/unexpected fields")
        concept = item["target_concept"]
        if concept not in targets or concept in seen_targets:
            raise ValueError(f"Unknown/duplicate target concept: {concept}")
        seen_targets.add(concept)
        if item["decision"] not in ("CREATE", "REUSE"):
            raise ValueError(f"Invalid decision for {concept}")
        matches = {x["iri"] for k in ("source_matches", "import_matches") for x in targets[concept][k]}
        analogues = {x["iri"] for x in targets[concept]["air_pattern_analogs"]}
        if not isinstance(item["analogue_iris"], list) or not set(item["analogue_iris"]).issubset(analogues):
            raise ValueError(f"Ungrounded analogue IRI for {concept}")
        if not isinstance(item["rationale"], str) or len(item["rationale"].strip()) < 12:
            raise ValueError(f"Missing rationale for {concept}")
        if not isinstance(item["review_notes"], list):
            raise ValueError(f"review_notes must be a list for {concept}")
        if matches:
            if item["decision"] != "REUSE" or item["reused_iri"] not in matches:
                raise ValueError(f"Expected REUSE of exact local match for {concept}")
            if any(item[f] for f in ("local_name", "parent_iri", "definition")):
                raise ValueError(f"REUSE must not redefine {concept}")
        else:
            if item["decision"] != "CREATE" or item["reused_iri"]:
                raise ValueError(f"Expected CREATE for missing rail concept {concept}")
            name = item["local_name"]
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"Invalid local_name for {concept}: {name}")
            new_iri = namespace + name
            if new_iri in classes | properties | individuals | proposed_iris:
                raise ValueError(f"Duplicate or preexisting proposed IRI: {new_iri}")
            proposed_iris.add(new_iri)
            if len(item["definition"].strip()) < 15 or not item["label"].strip():
                raise ValueError(f"Missing label/definition for {concept}")
    if seen_targets != set(targets):
        raise ValueError("Missing pattern targets")

    allowed_classes = classes | proposed_iris
    if not response["object_properties"]:
        raise ValueError("At least one rail-domain object property proposal is required")
    for item in decisions:
        if item["decision"] == "CREATE":
            if item["parent_iri"] not in allowed_classes or item["parent_iri"] == namespace + item["local_name"]:
                raise ValueError(f"Ungrounded/self parent for {item['target_concept']}: {item['parent_iri']}")
            if item["target_concept"] == "rail passenger role" and item["parent_iri"] != BASE_TYPES["role"]:
                raise ValueError("Rail passenger role must directly specialize BFO Role in this stage")
            if item["target_concept"] == "rail journey" and item["parent_iri"] != BASE_TYPES["process"]:
                raise ValueError("Rail journey must directly specialize BFO Process in this stage")
    new_props = set()
    for key in ("object_properties", "data_properties"):
        for item in response[key]:
            spec = SCHEMA["properties"][key]["items"]["properties"]
            if not isinstance(item, dict) or set(item) != set(spec):
                raise ValueError(f"Invalid {key} item fields")
            name = item["local_name"]
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"Invalid property local_name: {name}")
            iri = namespace + name
            if iri in properties | classes | individuals | proposed_iris | new_props:
                raise ValueError(f"Duplicate or existing property IRI: {iri}")
            new_props.add(iri)
            if item["domain_iri"] not in allowed_classes:
                raise ValueError(f"Unknown domain for {iri}: {item['domain_iri']}")
            if len(item["definition"].strip()) < 12 or len(item["rationale"].strip()) < 12:
                raise ValueError(f"Missing property explanation for {iri}")
            if key == "object_properties":
                if item["range_iri"] not in allowed_classes:
                    raise ValueError(f"Unknown range for {iri}: {item['range_iri']}")
                if item["reused_analogue_iri"] and item["reused_analogue_iri"] not in properties:
                    raise ValueError(f"Unknown reused analogue property for {iri}")
            elif item["datatype_iri"] not in {
                "http://www.w3.org/2001/XMLSchema#string", "http://www.w3.org/2001/XMLSchema#dateTime",
                "http://www.w3.org/2001/XMLSchema#integer", "http://www.w3.org/2001/XMLSchema#decimal",
                "http://www.w3.org/2001/XMLSchema#date", "http://www.w3.org/2001/XMLSchema#boolean"}:
                raise ValueError(f"Unsupported data property range for {iri}")
    if not all(isinstance(x, str) for x in response["design_warnings"]):
        raise ValueError("design_warnings contains non-text values")
    return {"created_classes": len(proposed_iris), "reused_classes": len(decisions) - len(proposed_iris),
            "object_properties": len(response["object_properties"]),
            "data_properties": len(response["data_properties"])}


def call_llm(payload: dict, model: str) -> dict:
    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("XAI_API_KEY missing. Set it only in your current PowerShell session before running live mode.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI Python SDK: python -m pip install openai") from exc
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1", timeout=900.0, max_retries=2)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)},
        ],
        response_format={"type": "json_schema", "json_schema": {
            "name": "rail_tbox_design_proposal", "strict": True, "schema": SCHEMA}},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned no content")
    return json.loads(content)


def run(root: Path, *, model: str, namespace: str, dry_run: bool = False,
        llm_function=None) -> dict:
    source = root / "src/module.ttl"
    original_hash = sha256_file(source)
    payload, classes, properties, individuals = build_request(root, namespace, model)
    if sha256_file(source) != original_hash:
        raise RuntimeError("Source ontology changed while preparing request")
    # Keep input manifest for reproducible prompt inspection and future pipeline reuse.
    atomic_json(root / "generated/rail_design_request.json", payload)
    if dry_run:
        result = {"status": "DRY_RUN_PASS", "model": model, "source_unchanged": True,
                  "targets": len(payload["pattern_targets"]), "imports": len(payload["local_imports"]),
                  "note": "No API request made; no OWL generated"}
    else:
        llm_function = llm_function or call_llm
        response = llm_function(payload, model)

        # Preserve the original LLM response for review.
        atomic_json(
            root / "generated/rail_pattern_proposal_raw.json",
            response
        )

        # Enforce mandatory BFO parent classes.
        required_parents = {
            "rail passenger role": BASE_TYPES["role"],
            "rail journey": BASE_TYPES["process"],
        }

        for item in response.get("rail_classes", []):
            concept = item.get("target_concept")

            if (
                item.get("decision") == "CREATE"
                and concept in required_parents
            ):
                expected = required_parents[concept]

                if item.get("parent_iri") != expected:
                    original = item.get("parent_iri")
                    item["parent_iri"] = expected

                    item["review_notes"].append(
                        f"Pipeline constraint: parent changed "
                        f"from {original} to {expected}"
                    )

        counts = validate_proposal(
            response,
            payload,
            classes,
            properties,
            individuals
        )
        if sha256_file(source) != original_hash:
            raise RuntimeError("Source ontology changed during generation; aborting")
        proposal = {"status": "PROPOSED_NOT_VALIDATED_OWL", "script_version": VERSION,
                    "provenance": {"model": model, "source_sha256": original_hash,
                                   "prompt_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                                   "local_import_hashes": {k: v["sha256"] for k, v in payload["local_imports"].items()},
                                   "generated_by": "xAI LLM; checked against local RDF vocabulary"},
                    "namespace": namespace, "proposal": response}
        atomic_json(root / "generated/rail_pattern_proposal.json", proposal)
        result = {"status": "PASS_PROPOSAL_ONLY", "model": model,
                  "source_unchanged": True, "targets": len(payload["pattern_targets"]),
                  "imports": len(payload["local_imports"]), **counts,
                  "note": "JSON candidate design only; no OWL generated or merged"}
    atomic_json(root / "reports/step2_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", default=os.getenv("XAI_MODEL", "grok-4.7"))
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--dry-run", action="store_true", help="Test input/import checks without an API call")
    args = parser.parse_args()
    try:
        result = run(args.project_root.resolve(), model=args.model, namespace=args.namespace,
                     dry_run=args.dry_run)
    except Exception as exc:
        print(f"STEP 2 — FAIL: {exc}", file=sys.stderr)
        return 1
    print("STEP 2 —", result["status"])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
