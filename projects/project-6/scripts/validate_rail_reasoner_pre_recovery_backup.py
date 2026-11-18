#!/usr/bin/env python3
"""Project 6 Step 6: baseline vs refined rail candidate OWL reasoning with ROBOT.

Run in project-6 root, after successful Step 5C:
    python scripts/validate_rail_reasoner.py

Explicitly no ontology merge, automatic semantic approval, new individuals, or LLM call.
Runs ROBOT/HermiT twice over a complete *locally supplied* flattened corpus,
without resolving remote owl:imports. Reports scope of that import closure.
ROBOT exit 0 is required for both consistency and no unsatisfiable named classes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from rdflib import BNode, Graph, URIRef
from rdflib.compare import to_canonical_graph, isomorphic
from rdflib.namespace import RDF, RDFS, OWL

from prepare_rail_mowl import confirm_step4, load_ontology
from generate_rail_axioms import SUPPORTED_EXTENSIONS
from validate_rail_tbox import atomic_json, atomic_text, file_hash

VERSION = "project6-robot-reasoner-v1"
STEP6_ONTOLOGY = URIRef("https://example.org/ontologies/project6-step6-local-flattened")


def _stable_graph(g: Graph, namespace_key: str) -> Graph:
    """Canonicalize *each input separately* and avoid cross-file blank node collisions."""
    result = Graph()
    prefix = "g" + hashlib.sha256(namespace_key.encode("utf-8")).hexdigest()[:10] + "_"
    for s, p, o in to_canonical_graph(g):
        result.add((BNode(prefix + str(s)) if isinstance(s, BNode) else s,
                    BNode(prefix + str(p)) if isinstance(p, BNode) else p,
                    BNode(prefix + str(o)) if isinstance(o, BNode) else o))
    return result


def _serialize(g: Graph) -> str:
    """Deterministic fully expanded Turtle including locally stable blank node IDs."""
    return ("# Project 6 Step 6: locally flattened, reasoner-only graph; NOT source ontology.\n"
            "# All owl:imports resolved by local graph assembly, not by network fetch.\n\n"
            + "\n".join(sorted(f"{s.n3()} {p.n3()} {o.n3()} ." for s, p, o in g)) + "\n")


def _assemble(parts: list[tuple[str, Graph]], *, include_candidate: bool) -> tuple[Graph, dict]:
    union = Graph()
    original_imports = set()
    stripped_ontology_subjects = set()
    input_counts = {}
    for key, original in parts:
        stable = _stable_graph(original, key)
        input_counts[key] = len(stable)
        original_imports.update(str(o) for o in stable.objects(None, OWL.imports))
        ontology_subjects = set(stable.subjects(RDF.type, OWL.Ontology))
        stripped_ontology_subjects.update(str(x) for x in ontology_subjects)
        # OWL ontology headers/import declarations are document metadata, not domain axioms.
        # Remove to make the union one offline OWL ontology, not an attempted remote import.
        for s, p, o in stable:
            if p == OWL.imports or s in ontology_subjects:
                continue
            union.add((s, p, o))
    union.add((STEP6_ONTOLOGY, RDF.type, OWL.Ontology))
    report = {"input_triples": input_counts,
              "original_owl_import_iris_suppressed_for_offline_flattening": sorted(original_imports),
              "original_ontology_header_iris_suppressed": sorted(stripped_ontology_subjects),
              "flattened_triples": len(union),
              "candidate_included": include_candidate,
              "import_closure_scope": "Only six hash-verified local import files; no remote owl:imports fetched."
              }
    return union, report


def _named_pairs(graph: Graph) -> set[tuple[str, str]]:
    return {(str(s), str(o)) for s, o in graph.subject_objects(RDFS.subClassOf)
            if isinstance(s, URIRef) and isinstance(o, URIRef) and s != o}


def _named_individuals(graph: Graph) -> set[str]:
    return {str(s) for s in graph.subjects(RDF.type, OWL.NamedIndividual) if isinstance(s, URIRef)}


def _verify(root: Path) -> dict:
    source, refined, proposal, audit, step4, hashes = confirm_step4(root)
    summary_path = root / "reports/step5_v2_summary.json"
    review_path = root / "reports/step5_v2_grounded_review.json"
    if not summary_path.is_file() or not review_path.is_file():
        raise FileNotFoundError("Step 5C grounded review reports missing; run Step 5C first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (summary.get("status") != "PASS_GROUNDED_HYBRID_REVIEW_ONLY"
        or summary.get("llm_executed") is not True
        or summary.get("semantic_approval") is not False
        or summary.get("source_unchanged") is not True
        or summary.get("merged_into_source") is not False
        or summary.get("owl_reasoner_executed") is not False
        or summary.get("source_sha256") != hashes["source"]
        or review.get("refined_sha256") != hashes["refined"]
        or review.get("source_sha256") != hashes["source"]
        or review.get("no_automated_approval") is not True
        or len(review.get("candidate_relationships", [])) != 9
        or any(r.get("approved") is not False for r in review["candidate_relationships"])):
        raise ValueError("Step 5C provenance/status is not an unapproved completed review")
    files = {
        "source": root / "src/module.ttl",
        "refined": root / "generated/rail_axioms_refined.ttl",
        "step4_summary": root / "reports/step4_summary.json",
        "step5c_summary": summary_path,
        "step5c_review": review_path,
    }
    import_paths = sorted(p for p in (root / "imports").rglob("*")
                          if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    expected_imports = proposal.get("provenance", {}).get("local_import_hashes", {})
    actual_imports = {"imports/" + p.relative_to(root / "imports").as_posix(): file_hash(p)
                      for p in import_paths}
    if not expected_imports or actual_imports != expected_imports:
        raise ValueError("Local import set or SHA-256 changed since Step 2")
    for p in import_paths:
        files["imports/" + p.relative_to(root / "imports").as_posix()] = p
    parts = [("src/module.ttl", source)]
    for p in import_paths:
        parts.append(("imports/" + p.relative_to(root / "imports").as_posix(), load_ontology(p)))
    if any((s, RDF.type, OWL.NamedIndividual) in refined for s in refined.subjects(RDF.type, OWL.NamedIndividual)):
        raise ValueError("Candidate added explicit named individuals")
    if len(refined) != 63:
        raise ValueError("Step 3A refined candidate was unexpectedly changed")
    all_hashes = {name: file_hash(path) for name, path in files.items()}
    return {"source": source, "refined": refined, "parts": parts,
            "paths": files, "hashes": all_hashes, "review": review,
            "step4": step4, "audit": audit}


def _robot_command(robot: str | None, robot_jar: Path | None, java: str) -> list[str]:
    if robot_jar is not None:
        if not robot_jar.is_file():
            raise FileNotFoundError(f"ROBOT JAR not found: {robot_jar}")
        if not shutil.which(java) and not Path(java).is_file():
            raise RuntimeError("Java executable not found: " + java)
        return [java, "-jar", str(robot_jar.resolve())]
    resolved = shutil.which(robot or "robot")
    if resolved is None:
        raise FileNotFoundError("ROBOT executable not found on PATH; supply --robot-jar C:\\path\\robot.jar")
    return [resolved]


def _reason(base_cmd: list[str], source: Path, output: Path, log: Path,
            reasoner: str, timeout: int) -> tuple[bool, dict]:
    cmd = base_cmd + ["reason", "--input", str(source), "--reasoner", reasoner,
                      "--output", str(output)]
    output.unlink(missing_ok=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                                timeout=timeout, check=False)
        status = result.returncode
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        status = None
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "") + f"\nROBOT timed out after {timeout} seconds"
    except OSError as exc:
        status = None
        stdout, stderr = "", f"Failed to start ROBOT: {exc}"
    atomic_text(log, "COMMAND: " + " ".join(cmd) + "\nRETURN CODE: " + str(status)
                + "\n\nSTDOUT:\n" + stdout + "\nSTDERR:\n" + stderr + "\n")
    ok = status == 0 and output.is_file() and output.stat().st_size > 0
    return ok, {"return_code": status, "log": log.name,
                "reasoned_file": output.name if ok else None,
                "error_summary": None if ok else (stderr + "\n" + stdout)[-1800:],
                "executed": status is not None}


def _check_unchanged(context: dict) -> None:
    for label, file in context["paths"].items():
        if file_hash(file) != context["hashes"][label]:
            raise RuntimeError("Protected upstream input changed during Step 6: " + label)


def validate(root: Path, *, robot: str | None = None, robot_jar: Path | None = None,
             java: str = "java", reasoner: str = "hermit", timeout: int = 900) -> dict:
    root = root.resolve()
    if timeout < 30:
        raise ValueError("timeout must be at least 30 seconds")
    ctx = _verify(root)
    base_cmd = _robot_command(robot, robot_jar, java)
    gen = root / "generated"
    reports = root / "reports"
    gen.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    baseline, base_meta = _assemble(ctx["parts"], include_candidate=False)
    augmented, aug_meta = _assemble(ctx["parts"] + [("generated/rail_axioms_refined.ttl", ctx["refined"])],
                                     include_candidate=True)
    if not set(baseline).issubset(set(augmented)):
        raise ValueError("Flat baseline axioms lost in augmented ontology")
    new_triples = set(augmented) - set(baseline)
    if new_triples != set(ctx["refined"]):
        raise ValueError("Augmented logical graph does not match exactly the refined candidate delta")
    if _named_individuals(augmented) != _named_individuals(baseline):
        raise ValueError("Refined candidate added named individuals")
    baseline_path = gen / "step6_baseline_flattened.ttl"
    augmented_path = gen / "step6_candidate_flattened.ttl"
    atomic_text(baseline_path, _serialize(baseline))
    atomic_text(augmented_path, _serialize(augmented))
    if not isomorphic(load_ontology(baseline_path), baseline) or not isomorphic(load_ontology(augmented_path), augmented):
        raise RuntimeError("Step 6 local flattening failed graph-equivalent round trip")
    detail = {"version": VERSION, "reasoner": reasoner, "baseline": base_meta,
              "candidate": aug_meta,
              "new_candidate_triples": len(new_triples),
              "new_explicit_named_subclass_pairs": len(_named_pairs(ctx["refined"])),
              "local_input_hashes": ctx["hashes"],
              "baseline_flattened_sha256": file_hash(baseline_path),
              "augmented_flattened_sha256": file_hash(augmented_path),
              "scope_limits": ["Offline reasoning only uses the six supplied local imports; remote imports are not fetched.",
                               "ROBOT coherence checks do not determine whether modeling choices are ontologically appropriate.",
                               "Reported new named-pair comparisons use ROBOT default direct inferred superclasses; they are not a complete entailment difference.",
                               "Rail Ticket physical-vs-digital and Railway Station facility-vs-site await human resolution."],
              "merged_into_source": False,
              "semantic_approval": False}
    base_output = gen / "step6_baseline_reasoned.ttl"
    aug_output = gen / "step6_candidate_reasoned.ttl"
    ok_base, details_base = _reason(base_cmd, baseline_path, base_output,
                                     reports / "step6_baseline_robot.log", reasoner, timeout)
    detail["baseline_reasoner_run"] = details_base
    if ok_base:
        try:
            base_reasoned = load_ontology(base_output)
            detail["baseline_reasoned_named_pairs"] = len(_named_pairs(base_reasoned))
        except Exception as e:
            ok_base = False
            detail["baseline_reasoner_run"]["parse_error"] = str(e)
    if not ok_base:
        status = "FAIL_OR_BLOCKED_BASELINE_REASONER"
    else:
        ok_aug, details_aug = _reason(base_cmd, augmented_path, aug_output,
                                      reports / "step6_candidate_robot.log", reasoner, timeout)
        detail["augmented_reasoner_run"] = details_aug
        if ok_aug:
            try:
                aug_reasoned = load_ontology(aug_output)
                direct_candidate = _named_pairs(ctx["refined"])
                new_reasoned = _named_pairs(aug_reasoned) - _named_pairs(base_reasoned)
                newly_inferred = new_reasoned - direct_candidate
                detail["augmented_reasoned_named_pairs"] = len(_named_pairs(aug_reasoned))
                detail["new_named_pairs_vs_baseline_reasoned"] = len(new_reasoned)
                detail["new_named_pairs_not_explicit_in_candidate"] = len(newly_inferred)
                detail["new_named_pairs_examples"] = [{"child": a, "parent": b} for a, b in sorted(new_reasoned)[:100]]
                detail["new_reasoner_only_pairs_examples"] = [{"child": a, "parent": b} for a, b in sorted(newly_inferred)[:100]]
            except Exception as e:
                ok_aug = False
                detail["augmented_reasoner_run"]["parse_error"] = str(e)
        status = "PASS_REASONER_COHERENT_UNAPPROVED" if ok_aug else "FAIL_AUGMENTED_REASONER"
    _check_unchanged(ctx)
    detail["status"] = status
    atomic_json(reports / "step6_reasoner_detail.json", detail)
    summary = {"version": VERSION, "status": status,
               "baseline_reasoner_passed": ok_base,
               "augmented_reasoner_passed": status == "PASS_REASONER_COHERENT_UNAPPROVED",
               "owl_reasoner_executed": bool(detail["baseline_reasoner_run"].get("executed")),
               "source_unchanged": True,
               "source_sha256": ctx["hashes"]["source"],
               "refined_sha256": ctx["hashes"]["refined"],
               "source_merged": False,
               "new_named_individuals": 0,
               "semantic_approval": False,
               "pending_scope_review": ["rail ticket physical bearer vs digital information",
                                        "railway station physical facility vs site"],
               "next": "Resolve human scope decisions, then controlled candidate merge and full integrated pipeline"}
    atomic_json(reports / "step6_reasoner_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--robot", help="ROBOT CLI executable (default: search PATH)")
    parser.add_argument("--robot-jar", type=Path, help="Use java -jar PATH instead of the ROBOT CLI")
    parser.add_argument("--java", default="java")
    parser.add_argument("--reasoner", default="hermit", help="Default hermit (OWL DL)")
    parser.add_argument("--timeout", type=int, default=900, help="Seconds per reasoner invocation")
    args = parser.parse_args()
    try:
        summary = validate(args.project_root, robot=args.robot, robot_jar=args.robot_jar,
                           java=args.java, reasoner=args.reasoner, timeout=args.timeout)
        print("STEP 6 — " + summary["status"])
        return 0 if summary["status"] == "PASS_REASONER_COHERENT_UNAPPROVED" else 1
    except Exception as e:
        print("STEP 6 — FAIL: " + str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
