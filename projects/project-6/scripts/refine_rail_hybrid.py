"""Project 6 Step 5C: source-grounded LLM review + honest joint-evidence reporting.

Run from the project root after successful Step 5 (200 epochs):
    python scripts/refine_rail_hybrid.py --prepare-only
    python scripts/refine_rail_hybrid.py --with-llm

Uses the VERIFIED source and local imports, and reuses existing mOWL vectors.
No model training, source modification, automatic axiom acceptance, or merge.
Original Step 5 reports are NOT overwritten.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDFS, SKOS

import evaluate_rail_mowl as old
import prepare_rail_mowl as pre

VERSION = "project6-elembeddings-grounded-review-v2"
BFO_ROLE = "http://purl.obolibrary.org/obo/BFO_0000023"


def triples_from(*graphs: Graph) -> Graph:
    out = Graph()
    for graph in graphs:
        for triple in graph:
            out.add(triple)
    return out


def iri_label(graph: Graph, iri: str) -> str:
    return old.source_label(graph, iri)


def definitions(graph: Graph, iri: str) -> list[str]:
    # SKOS definition used by refined candidate, IAO_0000115 used by imports.
    iri_ref = URIRef(iri)
    predicates = (SKOS.definition, URIRef("http://purl.obolibrary.org/obo/IAO_0000115"))
    return sorted({str(obj)[:1500] for pred in predicates for obj in graph.objects(iri_ref, pred)})[:3]


def direct_parents(graph: Graph, iri: str) -> list[str]:
    return sorted({str(o) for o in graph.objects(URIRef(iri), RDFS.subClassOf)
                   if isinstance(o, URIRef)})


def verified_ancestry(graph: Graph, start: str, target: str, *, max_hops: int = 16) -> list[str] | None:
    """Find an explicitly ASSERTED subclass path; no reasoner/inferences claimed."""
    if start == target:
        return [start]
    q = deque([(start, [start])])
    seen = {start}
    while q:
        node, path = q.popleft()
        if len(path) > max_hops:
            continue
        for parent in direct_parents(graph, node):
            if parent == target:
                return path + [target]
            if parent not in seen:
                seen.add(parent)
                q.append((parent, path + [parent]))
    return None


def _verified_parent_facts(source: Graph, imports: Graph, parent: str) -> list[dict]:
    out = []
    for parent_of_parent in direct_parents(triples_from(source, imports), parent):
        locations = []
        statement = (URIRef(parent), RDFS.subClassOf, URIRef(parent_of_parent))
        if statement in source:
            locations.append("SOURCE_MODULE")
        if statement in imports:
            locations.append("VERIFIED_LOCAL_IMPORT")
        out.append({"subject": parent, "predicate": str(RDFS.subClassOf),
                    "object": parent_of_parent, "asserted_in": locations,
                    "object_label": iri_label(triples_from(source, imports), parent_of_parent),
                    "verification": "DIRECTLY_ASSERTED_NOT_INFERRED"})
    return out


def grounded_request(root: Path, split: dict, model: str) -> dict:
    source, candidate, proposal, _audit, _step4, hashes = pre.confirm_step4(root)
    imported, import_hashes = pre.collect_imports(root, proposal)
    established = triples_from(source, imported)
    context = triples_from(established, candidate)
    requests = []
    for i, pair in enumerate(split["candidate_pairs"]):
        child, parent = pair["child"], pair["parent"]
        parent_role_path = verified_ancestry(established, parent, BFO_ROLE)
        child_role_path = verified_ancestry(established, child, BFO_ROLE)
        other_candidate_parents = [p for p in direct_parents(candidate, child) if p != parent]
        requests.append({
            "id": f"edge-{i:02d}", "child_iri": child, "parent_iri": parent,
            "child_label": iri_label(context, child),
            "parent_label": iri_label(context, parent),
            "child_definitions": definitions(context, child),
            "parent_definitions": definitions(established, parent),
            "verified_parent_direct_superclasses": _verified_parent_facts(source, imported, parent),
            "verified_parent_is_bfo_role_or_subclass": parent_role_path is not None,
            "verified_parent_role_ancestry_path": parent_role_path,
            "verified_child_role_ancestry_path": child_role_path,
            "other_proposed_child_parents_NOT_YET_APPROVED": other_candidate_parents,
            "cosine_evidence_type": pair["evaluation"],
            "design_note": "Rail ticket as physical bearer vs electronic ticket content is UNRESOLVED" if child.endswith("#rail_ticket") else "",
        })
    return {
        "version": VERSION, "model": model,
        "source_sha256": hashes["source"], "refined_sha256": hashes["refined"],
        "import_hashes": import_hashes,
        "candidates": requests,
        "ontology_context_rules": [
            "All VERIFIED_PARENT facts and role-ancestry paths come from the unchanged source and hash-verified local imports.",
            "Other candidate parents are PROPOSALS, not asserted source facts or reasoner results.",
            "When source facts say a parent is a BFO Role subclass, do NOT hypothesize that it is a person/bearer instead.",
            "Score plausibility of the proposed subclass relation, not whether the axiom has already been approved.",
            "If facts are insufficient, say exactly what is missing rather than inventing a parent's type.",
            "No people, individuals, intent, or threat identification; no merges or automatic acceptance."
        ]
    }


def call_grounded_llm(request: dict) -> dict:
    """Use a source-aware instruction instead of the original label-only instruction."""
    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("XAI_API_KEY missing; set it securely in this PowerShell session")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai Python package is required in .venv-mowl") from exc
    schema = {"type": "object", "additionalProperties": False,
              "required": ["ratings"], "properties": {"ratings": {"type": "array", "items": {
              "type": "object", "additionalProperties": False,
              "required": ["id", "plausibility", "rationale", "concern"],
              "properties": {"id": {"type": "string"}, "plausibility": {"type": "number"},
                             "rationale": {"type": "string"}, "concern": {"type": "string"}}}}}}
    system = (
        "You are reviewing proposed OWL named-subclass relationships for a BFO/IAO academic ontology. "
        "Read the hash-verified, DIRECTLY ASSERTED source/import superclass axioms in each candidate "
        "BEFORE evaluating the proposed subclass. Do not contradict the verified parent hierarchy. "
        "If an asserted source axiom places a parent under BFO_0000023 (Role), never call it a person, "
        "a bearer, or an untyped passenger. Distinguish the class's role TYPE from who bears the role. "
        "Proposed candidate superclass axioms are NOT verified source facts. "
        "The verified type of a parent is not, by itself, proof that the proposed child is its subclass. "
        "Assess the genus-differentia definitions and compatibility of each child-parent relation. "
        "Score 0..1 as a provisional plausibility rating, not a probability or logical entailment. "
        "Point out physical versus information bearer scope uncertainty for Rail Ticket. "
        "Return exactly one JSON rating for every id, with fields id, plausibility, rationale, concern. "
        "Do not invent individuals, infer threats, accept axioms, or modify ontology files."
    )
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1", timeout=900.0, max_retries=0)
    reply = client.chat.completions.create(
        model=request["model"],
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": json.dumps(request, sort_keys=True)}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "rail_grounded_review_v2", "strict": True, "schema": schema}})
    data = reply.choices[0].message.content
    if not data:
        raise RuntimeError("xAI returned no response content")
    return json.loads(data)


def _check_v1(root: Path, split: dict) -> tuple[list[dict], dict, dict]:
    original_report = json.loads((root / "reports/step5_cosine_scores.json").read_text(encoding="utf-8"))
    original_review = json.loads((root / "reports/step5_hybrid_review.json").read_text(encoding="utf-8"))
    original_summary = json.loads((root / "reports/step5_summary.json").read_text(encoding="utf-8"))
    if original_summary.get("status") != "PASS_HYBRID_REVIEW_ONLY" or not original_summary.get("llm_executed"):
        raise ValueError("Step 5 v1 hybrid review must finish before Step 5C")
    if original_summary.get("source_sha256") != split["source_sha256"]:
        raise ValueError("Source hash changed since the original hybrid review")
    model = original_report.get("model", {})
    expected = {"version": old.VERSION, "train_sha256": split["train_sha256"],
                "valid_sha256": split["valid_sha256"], "source_sha256": split["source_sha256"],
                "refined_sha256": split["refined_sha256"], "seed": split["seed"],
                "epochs": model.get("epochs"), "embed_dim": model.get("embed_dim"),
                "device": "cpu", "model_type": "mOWL ELEmbeddings",
                "model_path": "generated/mowl_rail_train.ttl"}
    if expected["epochs"] != 200:
        raise ValueError("Step 5C requires the 200-epoch model for this run; found " + str(expected["epochs"]))
    for key, val in expected.items():
        if model.get(key) != val:
            raise ValueError(f"Step 5 cosine model metadata mismatch: {key}")
    index, vectors, _ = old.load_vectors(root, expected)
    scores = old.score_holdouts(index, vectors, split)
    old_scores = {(x["child"], x["parent"]): x for x in original_report["scores"]}
    if len(scores) != len(old_scores):
        raise ValueError("Original cosine reference scores incomplete")
    for row in scores:
        old_row = old_scores.get((row["child"], row["parent"]))
        if old_row is None or old_row["cosine"] != row["cosine"]:
            raise ValueError("Recomputed cached embedding scores disagree with original Step 5")
    return scores, original_review, original_report


def _reference_percentile(score: float, reference: list[float]) -> float | None:
    if not reference:
        return None
    # Descriptive rank only. Source/import reference positives are NOT negatives or calibration data.
    return round(sum(x <= score for x in reference) / len(reference), 6)


def joint_review(split: dict, scores: list[dict], original: dict,
                 ratings: dict[str,dict] | None, request: dict) -> dict:
    score_lookup = {(r["child"], r["parent"]): r for r in scores}
    previous = {(r["child"], r["parent"]): r for r in original["candidate_relationships"]}
    request_by_id = {r["id"]: r for r in request["candidates"]}
    reference = [s["cosine"] for s in scores if s["scope"] == "SOURCE_IMPORT_REFERENCE" and s["cosine"] is not None]
    rows = []
    for i, edge in enumerate(split["candidate_pairs"]):
        child, parent = edge["child"], edge["parent"]
        key = child, parent
        raw_cos = score_lookup.get(key, {})
        is_independent = edge["evaluation"] == "HELD_OUT" and raw_cos.get("status") == "HELD_OUT_COSINE"
        cosine = raw_cos.get("cosine") if is_independent else None
        req = request_by_id[f"edge-{i:02d}"]
        llm = ratings.get(f"edge-{i:02d}") if ratings is not None else None
        prior = previous.get(key, {})
        if not prior:
            raise ValueError("Step 5 v1 comparison relationship missing: " + str(key))
        flags = []
        if llm is not None and req["verified_parent_is_bfo_role_or_subclass"] and llm["plausibility"] < 0.5:
            flags.append("LOW_LLM_SCORE_DESPITE_VERIFIED_PARENT_ROLE_TYPE_REVIEW_RATIONALE")
        if child.endswith("#rail_ticket"):
            flags.append("TICKET_PHYSICAL_BEARER_VS_INFORMATION_UNRESOLVED")
        if child.endswith("#railway_station"):
            flags.append("STATION_FACILITY_VS_SITE_SCOPE_REVIEW")
        rows.append({
            "child": child, "parent": parent,
            "cosine": cosine,
            "cosine_evidence": "HELD_OUT_INDEPENDENT" if cosine is not None else "INSUFFICIENT_INDEPENDENT_COSINE_EVIDENCE",
            "cosine_reference_percentile_descriptive": _reference_percentile(cosine, reference) if cosine is not None else None,
            "llm_plausibility": llm["plausibility"] if llm else None,
            "llm_rationale": llm["rationale"] if llm else None,
            "llm_concern": llm["concern"] if llm else None,
            "previous_llm_plausibility": prior.get("llm_plausibility"),
            "previous_v1_heuristic_hybrid_score_UNCALIBRATED": prior.get("hybrid_score"),
            "verified_parent_role_ancestry_path": req["verified_parent_role_ancestry_path"],
            "has_both_evidence_types": cosine is not None and llm is not None,
            # A percentile among positive source pairs does not calibrate P(subclass true).
            "hybrid_confidence_score": None,
            "grounding_flags": flags,
            "approved": False,
            "human_review_required": True,
        })
    return {
        "version": VERSION,
        "status": "GROUNDED_JOINT_REVIEW_ONLY" if ratings is not None else "GROUNDED_REQUEST_PREPARED_ONLY",
        "method": "Separate held-out cosine and source-grounded Grok. No numeric hybrid probability until calibrated with independent positives AND negatives.",
        "source_reference_scores_count": len(reference),
        "joint_evidence_pairs": sum(x["has_both_evidence_types"] for x in rows),
        "llm_ratings_executed": ratings is not None,
        "no_fabricated_cosine": True,
        "no_automated_approval": True,
        "candidate_relationships": rows,
        "score_interpretation": {
            "cosine": "[-1,1] vector directional similarity only, not entailment or subclass confidence",
            "cosine_reference_percentile_descriptive": "Fraction of held-out source/import subclass pairs with cosine <= rail pair. NOT a validity probability; no negative-label control.",
            "llm_plausibility": "Uncalibrated heuristic supplied by Grok; must be checked against verified source facts",
            "hybrid_confidence_score": "Always null until a proper labeled calibration is created; the v1 weighted number is saved only for audit"
        },
        "unresolved_design_decisions": ["rail ticket physical bearer vs electronic information", "railway station facility vs site"],
        "merged_into_source": False, "owl_reasoner_executed": False,
    }


def execute(root: Path, *, model: str = old.DEFAULT_MODEL, with_llm: bool = False,
            llm_function=None) -> dict:
    root = root.resolve()
    split = pre.plan(root, seed=42)
    scores, previous_review, cosine_report = _check_v1(root, split)
    req = grounded_request(root, split, model)
    output = root / "reports"
    gen = root / "generated"
    old.atomic_json(gen / "step5_v2_grounded_llm_request.json", req)
    ratings = None
    if with_llm:
        fn = llm_function or call_grounded_llm
        raw = fn(req)
        ratings = old.validate_llm_reply(raw, req)
        # Raw LLM response retained for provenance, even if it contradicts source data.
        old.atomic_json(gen / "step5_v2_grounded_llm_raw.json", raw)
    report = joint_review(split, scores, previous_review, ratings, req)
    report["source_sha256"] = split["source_sha256"]
    report["refined_sha256"] = split["refined_sha256"]
    report["llm_model"] = model if with_llm else None
    report["input_report_hashes"] = {
        name: pre.file_hash(output / name)
        for name in ("step5_cosine_scores.json", "step5_hybrid_review.json", "step5_summary.json")
    }
    # Ensure the source and refined candidate files have not changed during network IO.
    if pre.file_hash(root / "src/module.ttl") != split["source_sha256"] or \
       pre.file_hash(root / "generated/rail_axioms_refined.ttl") != split["refined_sha256"]:
        raise RuntimeError("Input changed while reviewing Step 5C")
    status = "PASS_GROUNDED_HYBRID_REVIEW_ONLY" if with_llm else "PASS_GROUNDED_REQUEST_PREPARED"
    summary = {"version": VERSION, "status": status,
               "epoch_model_reused": cosine_report["model"]["epochs"],
               "mowl_training_reexecuted": False,
               "llm_executed": with_llm,
               "llm_evaluated_pairs": sum(x["llm_plausibility"] is not None for x in report["candidate_relationships"]),
               "joint_evidence_pairs": report["joint_evidence_pairs"],
               "hybrid_confidence_scores_generated": 0,
               "source_unchanged": True, "source_sha256": split["source_sha256"],
               "merged_into_source": False, "new_named_individuals": 0,
               "semantic_approval": False, "owl_reasoner_executed": False,
               "next": "Review v1 vs v2 Passenger role rating, resolve modeling scope, then run OWL reasoner"}
    old.atomic_json(output / "step5_v2_grounded_review.json", report)
    old.atomic_json(output / "step5_v2_summary.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true", help="Prepare and inspect grounded prompt WITHOUT calling xAI")
    mode.add_argument("--with-llm", action="store_true", help="Call xAI with source-grounded facts; reuses 200-epoch vectors")
    ap.add_argument("--model", default=os.getenv("XAI_MODEL", old.DEFAULT_MODEL))
    args = ap.parse_args()
    try:
        summary = execute(args.project_root, model=args.model, with_llm=args.with_llm)
    except Exception as exc:
        print("STEP 5C — FAIL: " + str(exc), file=sys.stderr)
        return 1
    print("STEP 5C — " + summary["status"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
