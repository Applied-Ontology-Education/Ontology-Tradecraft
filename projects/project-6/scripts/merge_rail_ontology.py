#!/usr/bin/env python3
"""Project 6 Step 7: verified, source-preserving rail TBox merge.

Run from project-6 root:
    python scripts/merge_rail_ontology.py

Creates generated/module_rail_merged.ttl; NEVER overwrites src/module.ttl.
This policy encodes two explicitly reviewed design commitments:
  (1) Rail Ticket is a physical material information bearer (existing ticket parent).
  (2) Railway Station is a material entity representing station facilities.

Requires successful Step 4, Step 5C, and source-scoped Step 6. Reruns
SPARQL/SHACL over the merged graph. Checks that the exact 63-triple delta
matches the prior source-scoped HermiT-tested candidate. Does not assume
Grok/cosine scores are semantic approval or claim a new HermiT run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from validate_rail_tbox import (
    atomic_json, atomic_text, create_queries, create_shapes, file_hash,
    parse_turtle,
)

VERSION = "project6-controlled-rail-merge-v1"
NS = "https://example.org/ontologies/intel-ontology#"
AGENT = "https://www.commoncoreontologies.org/AgentOntology/"
TICKET = URIRef(NS + "rail_ticket")
STATION = URIRef(NS + "railway_station")
TRAIN = URIRef(NS + "train")
ROLE = URIRef(NS + "rail_passenger_role")
BFO_MATERIAL = URIRef("http://purl.obolibrary.org/obo/BFO_0000040")
IAO_MATERIAL_BEARER = URIRef("http://purl.obolibrary.org/obo/IAO_0000178")
EXISTING_TICKET = URIRef(AGENT + "ticket")
EXISTING_PASSENGER = URIRef(AGENT + "passenger")
BFO_ROLE = URIRef("http://purl.obolibrary.org/obo/BFO_0000023")
OUTPUT = Path("generated/module_augmented.ttl")


def _read_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _required(root: Path) -> dict[str, Path]:
    rels = {
        "source": "src/module.ttl",
        "refined": "generated/rail_axioms_refined.ttl",
        "step4": "reports/step4_summary.json",
        "step5c": "reports/step5_v2_summary.json",
        "step5c_review": "reports/step5_v2_grounded_review.json",
        "step6": "reports/step6_reasoner_summary.json",
        "step6_detail": "reports/step6_reasoner_detail.json",
        "step6_baseline": "generated/step6_baseline_flattened.ttl",
        "step6_candidate": "generated/step6_candidate_flattened.ttl",
    }
    paths = {k: root / p for k, p in rels.items()}
    missing = [str(p.relative_to(root)) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Step 7 requires these upstream files: " + ", ".join(missing))
    return paths


def _verify_provenance(paths: dict[str, Path], hashes: dict[str, str]) -> dict:
    s4 = _read_json(paths["step4"])
    s5 = _read_json(paths["step5c"])
    r5 = _read_json(paths["step5c_review"])
    s6 = _read_json(paths["step6"])
    d6 = _read_json(paths["step6_detail"])

    if (s4.get("status") != "PASS_QC_AND_SHACL_UNAPPROVED"
        or s4.get("sparql_passed") is not True
        or s4.get("shacl", {}).get("conforms") is not True):
        raise ValueError("Step 4 SPARQL/SHACL not passed")
    if (s5.get("status") != "PASS_GROUNDED_HYBRID_REVIEW_ONLY"
        or s5.get("llm_executed") is not True
        or s5.get("semantic_approval") is not False
        or r5.get("no_automated_approval") is not True
        or len(r5.get("candidate_relationships", [])) != 9):
        raise ValueError("Step 5C grounded review provenance invalid")
    if (s6.get("status") != "PASS_REASONER_COHERENT_UNAPPROVED"
        or s6.get("baseline_reasoner_passed") is not True
        or s6.get("augmented_reasoner_passed") is not True
        or s6.get("owl_reasoner_executed") is not True
        or s6.get("source_unchanged") is not True
        or s6.get("source_merged") is not False
        or s6.get("reasoning_scope") != "SOURCE_PLUS_STAR_LOCALITY_MODULE_OF_IMPORTS"
        or d6.get("status") != s6["status"]
        or d6.get("reasoner") != "hermit"
        or d6.get("merged_into_source") is not False):
        raise ValueError("Step 6 source-scoped HermiT validation not passed")
    for label, report in (("Step 4", s4), ("Step 5C", s5),
                          ("Step 5C review", r5), ("Step 6", s6)):
        if report.get("source_sha256") != hashes["source"]:
            raise ValueError(f"{label} source hash differs from current source")
    for label, report in (("Step 4", s4), ("Step 5C review", r5), ("Step 6", s6)):
        if report.get("refined_sha256") != hashes["refined"]:
            raise ValueError(f"{label} refined hash differs from current candidate")
    for label in ("baseline", "augmented"):
        run = d6.get("baseline_reasoner_run" if label == "baseline" else "augmented_reasoner_run", {})
        if run.get("return_code") != 0 or run.get("timed_out") is not False:
            raise ValueError(f"Step 6 {label} reasoner did not pass")
    if (d6.get("baseline_flattened_sha256") != hashes["step6_baseline"]
        or d6.get("augmented_flattened_sha256") != hashes["step6_candidate"]
        or d6.get("new_candidate_triples") != 63
        or d6.get("baseline", {}).get("source_triples") != 421
        or d6.get("candidate", {}).get("source_triples") != 421):
        raise ValueError("Step 6 reasoner input provenance mismatch")
    if (d6.get("local_input_hashes", {}).get("source") != hashes["source"]
        or d6.get("local_input_hashes", {}).get("refined") != hashes["refined"]):
        raise ValueError("HermiT input source/candidate hashes mismatch")
    return {"step4": s4, "step5c": s5, "step5c_review": r5, "step6": s6, "step6_detail": d6}


def _verify_design(source: Graph, refined: Graph, review: dict) -> dict:
    if len(source) != 421 or len(refined) != 63:
        raise ValueError("Unexpected source/refined graph size: expected 421 + 63")
    if any(isinstance(x, BNode) for triple in refined for x in triple):
        raise ValueError("Refined rail graph contains blank nodes")
    if (set(refined) & set(source)):
        raise ValueError("Refined candidate overlaps with existing source triples")
    if any(True for _ in refined.subjects(RDF.type, OWL.NamedIndividual)):
        raise ValueError("New named individuals are prohibited in the rail extension")
    expected_classes = {URIRef(NS + n) for n in (
        "train", "passenger_train", "railway_station", "rail_journey",
        "rail_journey_plan", "rail_ticket", "rail_reservation_information",
        "rail_passenger_role")}
    classes = set(refined.subjects(RDF.type, OWL.Class))
    objprops = set(refined.subjects(RDF.type, OWL.ObjectProperty))
    dataprops = set(refined.subjects(RDF.type, OWL.DatatypeProperty))
    if (classes != expected_classes or len(objprops) != 4 or len(dataprops) != 2
        or any(not str(s).startswith(NS) for s in objprops | dataprops)):
        raise ValueError("Unexpected rail class/property inventory")
    named_pairs = {(str(a), str(b)) for a, b in refined.subject_objects(RDFS.subClassOf)
                   if isinstance(a, URIRef) and isinstance(b, URIRef)}
    reviewed_pairs = {(x.get("child"), x.get("parent")) for x in review["candidate_relationships"]}
    if named_pairs != reviewed_pairs or len(named_pairs) != 9:
        raise ValueError("Merged subclass relationships differ from Step 5C review")
    required = {
        (TICKET, RDFS.subClassOf, EXISTING_TICKET),
        (EXISTING_TICKET, RDFS.subClassOf, IAO_MATERIAL_BEARER),
        (STATION, RDFS.subClassOf, BFO_MATERIAL),
        (TRAIN, RDFS.subClassOf, BFO_MATERIAL),
        (ROLE, RDFS.subClassOf, EXISTING_PASSENGER),
        (EXISTING_PASSENGER, RDFS.subClassOf, BFO_ROLE),
    }
    union = Graph()
    union += source
    union += refined
    if not all(t in union for t in required):
        raise ValueError("Required physical-ticket/material-station/source-role axioms absent")
    def definition(term: URIRef) -> str:
        vals = list(refined.objects(term, SKOS.definition))
        if len(vals) != 1 or not isinstance(vals[0], Literal) or vals[0].language != "en":
            raise ValueError(f"Expected single English definition for {term}")
        return str(vals[0]).lower()
    ticket_def = definition(TICKET)
    station_def = definition(STATION)
    if not ("material information bearer" in ticket_def and
            "rail" in ticket_def):
        # The definition in Step 3A uses 'material information bearer', which
        # establishes the physical bearer interpretation through its OWL parent.
        raise ValueError("Ticket definition no longer matches reviewed bearer scope")
    if not ("material entity" in station_def and "physical railway facilities" in station_def):
        raise ValueError("Railway Station definition no longer denotes material facilities")
    return {
        "rail_ticket": "physical material information bearer; rail ticket subclass of existing ticket",
        "railway_station": "material entity comprising physical railway facilities",
        "electronic_ticket_information": "not introduced; separate future information content entity",
        "immaterial_station_site": "not introduced; separate future site concept",
        "passenger_role": "rail passenger role specializes existing passenger role",
    }


def _verify_reasoner_delta(refined: Graph, paths: dict[str, Path], d6: dict) -> None:
    baseline = parse_turtle(paths["step6_baseline"])
    candidate = parse_turtle(paths["step6_candidate"])
    if len(candidate) - len(baseline) != 63:
        raise ValueError("HermiT input size delta differs from refined 63 triples")
    if not set(refined).issubset(set(candidate)):
        raise ValueError("HermiT candidate graph does not contain all proposed triples")
    # A full graph-isomorphism comparison detects nontrivial alterations to
    # anonymous GCAs, which ordinary triple-count comparisons can miss.
    expected = Graph()
    expected += baseline
    expected += refined
    if not isomorphic(expected, candidate):
        raise ValueError("HermiT-validated candidate is not baseline + exact refined graph")
    if d6.get("candidate", {}).get("flattened_triples") != len(candidate):
        raise ValueError("HermiT candidate count differs from report")


def _run_merged_qc(merged: Graph, refined: Graph) -> dict:
    """Re-run merged-graph QC and SHACL without RDFLib anti-join instability.

    Step 4's two FILTER NOT EXISTS language-annotation queries can produce
    false positives after repeated evaluations on some RDFLib/pyparsing
    versions. Check those two equivalent conditions directly from RDF triples,
    while executing the other nine generated SPARQL QC queries unchanged.
    """
    query_results = []
    terms = (set(refined.subjects(RDF.type, OWL.Class))
             | set(refined.subjects(RDF.type, OWL.ObjectProperty))
             | set(refined.subjects(RDF.type, OWL.DatatypeProperty)))
    for slug, query in sorted(create_queries(refined).items()):
        if slug == "04_missing_english_label" or slug == "05_missing_english_definition":
            predicate = RDFS.label if slug.startswith("04") else SKOS.definition
            failed_terms = [term for term in terms if not any(
                isinstance(val, Literal) and val.language == "en" and str(val).strip()
                for val in merged.objects(term, predicate))]
            row_count = len(failed_terms)
            method = "RDFLib equivalent explicit annotation check"
        else:
            target = refined if slug == "10_no_named_individuals_in_candidate" else merged
            row_count = len(list(target.query(query)))
            method = "SPARQL"
        query_results.append({"check": slug, "passed": row_count == 0,
                              "failures": row_count, "method": method})
    if not all(x["passed"] for x in query_results):
        bad = [x["check"] for x in query_results if not x["passed"]]
        raise ValueError("Step 7 merged-graph structural QC failed: " + ", ".join(bad))
    try:
        from pyshacl import validate as shacl_validate
    except ImportError as exc:
        raise RuntimeError("pyshacl missing in active venv; install pyshacl") from exc
    conforms, _, report = shacl_validate(
        data_graph=merged, shacl_graph=create_shapes(refined),
        inference="none", advanced=False, allow_infos=False,
        allow_warnings=False, abort_on_first=False,
    )
    if not conforms:
        raise ValueError("Step 7 merged-graph SHACL validation failed: " + str(report)[:1500])
    return {"checks": query_results, "sparql_queries_executed": 9,
            "equivalent_annotation_checks_executed": 2,
            "all_11_structural_checks_passed": True,
            "sparql_passed": True,
            "shacl_conforms": True, "shacl_report": str(report)[:1000]}


def build(root: Path, *, check_only: bool = False) -> dict:
    root = root.resolve()
    paths = _required(root)
    hashes = {key: file_hash(path) for key, path in paths.items()}
    reports = _verify_provenance(paths, hashes)
    source = parse_turtle(paths["source"])
    refined = parse_turtle(paths["refined"])
    decisions = _verify_design(source, refined, reports["step5c_review"])
    _verify_reasoner_delta(refined, paths, reports["step6_detail"])

    merged = Graph()
    merged += source
    merged += refined
    if len(merged) != 484:
        raise ValueError("Merged RDF graph must contain exactly 484 triples")
    qc = _run_merged_qc(merged, refined)

    # Output keeps original source bytes intact (including @prefix, original
    # imports, and anonymous GCAs), followed by fully qualified candidate N3
    # statements, valid in Turtle independently of the source prefixes.
    source_bytes = paths["source"].read_bytes()
    source_text = source_bytes.decode("utf-8")
    appendix = ("\n\n# Project 6 Step 7: reviewed rail TBox extension (63 triples).\n"
                "# Existing source statements and owl:imports above are unchanged.\n"
                + "\n".join(sorted(f"{s.n3()} {p.n3()} {o.n3()} ." for s, p, o in refined))
                + "\n")
    merged_text = source_text + appendix
    reparsed = Graph().parse(data=merged_text, format="turtle")
    if not isomorphic(reparsed, merged):
        raise ValueError("Serialized merge is not RDF-isomorphic to source + refined")
    if not merged_text.encode("utf-8").startswith(source_bytes):
        raise ValueError("Original source text not preserved byte-for-byte")
    for key, path in paths.items():
        if file_hash(path) != hashes[key]:
            raise RuntimeError("Upstream input changed during Step 7: " + key)
    summary = {
        "version": VERSION,
        "status": "PASS_CONTROLLED_MERGE_VALIDATED_NOT_PROMOTED" if not check_only
                  else "PASS_CONTROLLED_MERGE_CHECK_ONLY",
        "source_triples": len(source), "added_rail_triples": len(refined),
        "merged_triples": len(merged), "new_named_individuals": 0,
        "source_sha256": hashes["source"], "refined_sha256": hashes["refined"],
        "source_unchanged": True, "source_overwritten": False,
        "output": OUTPUT.as_posix() if not check_only else None,
        "step6_source_scoped_hermit_passed": True,
        "step7_hermit_rerun": False,
        "step7_reuses_exact_step6_reasoner_candidate_delta": True,
        "sparql_passed": True, "shacl_conforms": True,
        "design_decisions": decisions,
        "merge_policy": "Add all 63 reviewed rail TBox triples to separate output; do not replace source",
        "final_promotion_to_source": False,
        "automatic_llm_semantic_approval": False,
        "next": "Step 8: integrate the fully reproducible pipeline into scripts/run_all.py",
    }
    if check_only:
        return summary
    output_path = root / OUTPUT
    atomic_text(output_path, merged_text)
    if not isomorphic(parse_turtle(output_path), merged):
        raise RuntimeError("Written merged file failed final graph isomorphism check")
    if file_hash(paths["source"]) != hashes["source"]:
        raise RuntimeError("Source changed during output writing")
    summary["merged_sha256"] = file_hash(output_path)
    audit = {
        "version": VERSION, "status": summary["status"],
        "provenance_hashes": hashes,
        "design_decisions": decisions,
        "exact_candidate_delta_triples": len(refined),
        "upstream_hermit_reasoner": "HermiT source-scoped baseline and candidate passed; same 63 triples",
        "step7_checks": qc,
        "source_preservation": "Original source text is an exact byte prefix of generated output",
        "output_sha256": summary["merged_sha256"],
        "notes": [
            "No unrelated imported axioms are copied into merged source.",
            "The original owl:imports declarations remain in output.",
            "The prior HermiT check covered only the STAR locality module, not all remote imports.",
            "Grok and cosine scores are recorded evidence, not semantic proof.",
            "No automatic changes to src/module.ttl or original named individuals.",
        ],
    }
    atomic_json(root / "reports/step7_merge_audit.json", audit)
    atomic_json(root / "reports/step7_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-only", action="store_true",
                        help="Validate prerequisite reports, modeling policy, and RDF/QC; do not write outputs")
    args = parser.parse_args()
    try:
        result = build(args.project_root, check_only=args.check_only)
        print("STEP 7 — " + result["status"])
        print(json.dumps({k: v for k, v in result.items() if k not in ("design_decisions",)}, indent=2))
        return 0
    except Exception as exc:
        print("STEP 7 — FAIL: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
