#!/usr/bin/env python3
"""Project 6 frozen-v9 OWL audit with explicit import-closure handling.

Named-v9 mode audits EXACT frozen training graphs, despite missing external
imports. Full-closure mode requires every recursively declared owl:imports IRI
to match an explicitly local OWL ontology or its versionIRI, and refuses to
pretend a partial set constitutes the full OWL import closure.

Run from projects/project-6:
  python -u scripts/audit_v9_owl_entailment_v2.py --scope named-only --prepare-only
  python -u scripts/audit_v9_owl_entailment_v2.py --scope named-only --jvm-gb 8
  python -u scripts/audit_v9_owl_entailment_v2.py --scope full-closure --prepare-only
  python -u scripts/audit_v9_owl_entailment_v2.py --scope full-closure --jvm-gb 8

ONLY writes isolated *_v2 reports and generated OWL files. No network calls,
no editing of the base source, six imports, project pipeline, or v9 results.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL
import audit_v9_owl_entailment as base

VERSION = "project6-v9-owl-audit-v2-strict-or-explicit-named"
OUT = Path("reports/experiments/rail_mowl_v9_owl_audit_v2")
GEN = Path("generated/experiments/rail_mowl_v9_owl_audit_v2")
SOURCE_FILES = ("src/module.ttl", "generated/rail_axioms_refined.ttl", *base.IMPORTS)
EXTENSIONS = {".owl", ".ttl", ".rdf", ".xml"}


def fail(reason: str) -> None:
    raise RuntimeError(reason)


def relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def owl_identity(graph: Graph) -> set[str]:
    ids: set[str] = set()
    for entity in graph.subjects(RDF.type, OWL.Ontology):
        if isinstance(entity, URIRef):
            ids.add(str(entity))
            for version in graph.objects(entity, OWL.versionIRI):
                if isinstance(version, URIRef):
                    ids.add(str(version))
    return ids


def imports_from(graph: Graph) -> set[str]:
    return {str(iri) for iri in graph.objects(None, OWL.imports) if isinstance(iri, URIRef)}


def local_dependencies(root: Path, import_dir: Path):
    if not import_dir.exists():
        return {}, []
    if not import_dir.is_dir():
        fail(f"Import dependency path is not a directory: {import_dir}")
    candidates = sorted(p for p in import_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if len(candidates) > 500:
        fail("Too many local dependency candidates (>500). Narrow --import-dir explicitly.")
    dependencies = {}
    for path in candidates:
        rel = relative(root, path)
        graph = Graph()
        if path.suffix.lower() in {".rdf", ".xml"}:
            graph.parse(str(path), format="xml")
        else:
            graph = base.load(path)
        ids = owl_identity(graph)
        if not ids:
            fail(f"Local dependency lacks owl:Ontology identity; cannot match imports: {rel}")
        dependencies[rel] = graph
    return dependencies, [relative(root, p) for p in candidates]


def resolve_closure(root: Path, import_dir: Path):
    """Iteratively select matching local dependencies; never synthesize or fetch them."""
    parts = {}
    for rel in SOURCE_FILES:
        base.log("Parsing " + rel)
        parts[rel] = base.load(root / rel)
    extra, available_files = local_dependencies(root, import_dir)
    candidates = {**parts, **extra}
    identities: dict[str, set[str]] = defaultdict(set)
    for name, graph in candidates.items():
        for iri in owl_identity(graph):
            identities[iri].add(name)

    selected = set(parts)
    missing: dict[str, set[str]] = defaultdict(set)
    ambiguous: dict[str, set[str]] = defaultdict(set)
    q = deque(sorted(selected))
    completed = set()
    while q:
        owner = q.popleft()
        if owner in completed:
            continue
        completed.add(owner)
        for iri in sorted(imports_from(candidates[owner])):
            providers = identities.get(iri, set())
            if len(providers) == 0:
                missing[iri].add(owner)
            elif len(providers) > 1:
                ambiguous[iri].update(providers)
            else:
                (provider,) = tuple(providers)
                if provider not in selected:
                    selected.add(provider)
                    q.append(provider)

    mapping = {
        "status": ("UNRESOLVED_OR_AMBIGUOUS" if missing or ambiguous else "COMPLETE_EXPLICIT_LOCAL_CLOSURE"),
        "import_dir": relative(root, import_dir),
        "local_candidate_files": available_files,
        "loaded_files": sorted(selected),
        "unresolved_distinct_iris": len(missing),
        "unresolved_iris": [{"iri": iri, "imported_by": sorted(owners)} for iri, owners in sorted(missing.items())],
        "ambiguous_iris": [{"iri": iri, "candidate_files": sorted(owners)} for iri, owners in sorted(ambiguous.items())],
        "loaded_sha256": {p: base.digest(root / p) for p in sorted(selected)},
        "note": "Full-context OWL reasoning must not be claimed complete if any import is missing/ambiguous. No network import is fetched.",
    }
    base.write_json(root / OUT / "import_dependency_report.json", mapping)
    if missing or ambiguous:
        return None, mapping
    # Check the import closure, including all recursively discovered dependencies.
    selected_parts = {rel: candidates[rel] for rel in sorted(selected)}
    unresolved = base.import_map_validation(selected_parts)["unresolved_imports"]
    if unresolved:
        fail("Internal import resolution mismatch: " + json.dumps(unresolved, indent=2))
    return selected_parts, mapping


def verified_named_graph(root: Path, prep: dict, fold_idx: int):
    meta = prep["folds"][fold_idx - 1]
    rel = meta["outer_training_path"]
    if base.digest(root / rel) != meta["outer_sha256"]:
        fail(f"Frozen v9 named graph hash mismatch in fold {fold_idx}")
    g = base.load(root / rel)
    edges = base.named(g)
    if len(edges) != meta["outer_training_pairs"]:
        fail(f"Frozen v9 edge count differs in fold {fold_idx}")
    if not base.ANCHORS.issubset(edges):
        fail(f"Approved v9 anchors absent from fold {fold_idx}")
    if set(base.FOLDS[fold_idx - 1]) & (edges | base.closure(edges)):
        fail(f"Named training graph leaks holdout in fold {fold_idx}")
    return g


def save_audit_graph(root: Path, fold_idx: int, scope: str, graph: Graph):
    path = root / GEN / f"fold_{fold_idx}_{scope}.owl"
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(path), format="xml")
    round_trip = Graph().parse(str(path), format="xml")
    if set(base.FOLDS[fold_idx - 1]) & (base.named(round_trip) | base.closure(base.named(round_trip))):
        fail(f"Serialized {scope} fold {fold_idx} contains a named subclass leakage path")
    return {"path": relative(root, path), "sha256": base.digest(path), "triples": len(graph),
            "named_subclass_pairs": len(base.named(graph))}


def prepare(root: Path, scope: str, import_dir: Path):
    original_prep, summary = base.check_provenance(root)
    info = {"version": VERSION, "scope": scope, "status": "INCOMPLETE", "reasoner_run": False,
            "full_owl_semantic_entailment_checked": False,
            "v9_inputs_unmodified": True, "v9_native_auc": summary["native_auc"],
            "v9_native_mean": summary["native_8_class_mean"], "folds": []}
    parts = None
    mapping = None
    if scope == "full-closure":
        parts, mapping = resolve_closure(root, import_dir)
        info["import_dependency_report"] = relative(root, root / OUT / "import_dependency_report.json")
        info["import_resolution_status"] = mapping["status"]
        if parts is None:
            info["status"] = "BLOCKED_MISSING_LOCAL_OWL_IMPORTS"
            base.write_json(root / OUT / "full_closure_preparation.json", info)
            return info
        # Check exact authored source+refined augmented ontology separately from imports.
        augmented_check = base.check_augment_exact(root, parts)
        if not augmented_check["isomorphic_to_augmented"]:
            fail("Augmented ontology is not isomorphic to source + refined. Stop and review source provenance.")
        info["source_augmented_crosscheck"] = augmented_check
    for fold in range(1, 5):
        train_graph = verified_named_graph(root, original_prep, fold)
        graph = train_graph
        extra_meta = {}
        if scope == "full-closure":
            graph, extra_meta = base.build_full_graph(parts, fold)
            missing = base.named(train_graph) - base.named(graph)
            if missing:
                fail(f"Fold {fold} full closure omits {len(missing)} exact v9 named training edges")
        file_meta = save_audit_graph(root, fold, scope, graph)
        info["folds"].append({"fold": fold, "holdouts": [
            {"child": child, "parent": parent} for child, parent in base.FOLDS[fold - 1]],
            "dataset": {**file_meta, **extra_meta}})
    info["status"] = "PREPARED_NOT_REASONED"
    info["mode_description"] = (
        "Exact frozen v9 named-subclass fold training graphs only, not the full OWL import closure"
        if scope == "named-only" else
        "Source + refined rail TBox + six bundled local imports + explicitly resolved recursive import closure + approved anchors, minus fold targets"
    )
    file = root / OUT / ("named_only_preparation.json" if scope == "named-only" else "full_closure_preparation.json")
    base.write_json(file, info)
    return info


def reason(root: Path, prep: dict, jvm_gb: int):
    scope = prep["scope"]
    if prep["status"] != "PREPARED_NOT_REASONED":
        fail("Cannot invoke HermiT: " + prep["status"])
    report = {"version": VERSION, "scope": scope, "status": "INCOMPLETE", "reasoner": "HermiT",
              "full_owl_semantic_entailment_checked": False, "folds": []}
    target_file = root / OUT / ("named_only_entailment_report.json" if scope == "named-only" else "full_closure_entailment_report.json")
    try:
        bad = []
        for fold in prep["folds"]:
            idx = fold["fold"]
            fname = fold["dataset"]["path"]
            if base.digest(root / fname) != fold["dataset"]["sha256"]:
                fail(f"Prepared audit fold {idx} changed since preparation")
            base.log(f"HermiT {scope}, fold {idx}/4")
            result = base.reason_with_hermit(root / fname, [(x["child"], x["parent"]) for x in fold["holdouts"]], jvm_gb)
            report["folds"].append({"fold": idx, "reasoner_result": result})
            if not result["consistent"]:
                bad.append({"fold": idx, "reason": "INCONSISTENT_ONTOLOGY"})
            for case in result["results"]:
                if case["status"] != "NOT_ENTAILED":
                    bad.append({"fold": idx, **case})
            base.write_json(target_file, report)
        report["violations"] = bad
        report["status"] = (
            "FAIL_ENTAILED_OR_UNSATISFIABLE" if bad else
            "PASS_HERMIT_NAMED_CONTEXT_ONLY_FULL_CLOSURE_PENDING" if scope == "named-only" else
            "PASS_HERMIT_COMPLETE_LOCAL_IMPORT_CLOSURE"
        )
        report["full_owl_semantic_entailment_checked"] = scope == "full-closure"
        report["scope_limitation"] = (
            "This only validates v9 named-subclass training graphs; full imported OWL semantics NOT checked."
            if scope == "named-only" else
            "Full context was constructed from verified local import closure; model v9 itself trained on named subclasses only."
        )
    except Exception as ex:
        report["status"] = "ERROR_INCOMPLETE_DO_NOT_CLAIM_VALIDATED"
        report["error"] = str(ex)
        base.write_json(target_file, report)
        raise
    base.write_json(target_file, report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--scope", choices=("named-only", "full-closure"), default="named-only")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--import-dir", type=Path, default=Path("imports/audit_dependencies"))
    ap.add_argument("--jvm-gb", type=int, default=8)
    args = ap.parse_args()
    root = args.project_root.resolve()
    dep_dir = args.import_dir if args.import_dir.is_absolute() else root / args.import_dir
    if not dep_dir.resolve().is_relative_to(root):
        fail("Dependencies must reside within project root so paths are auditable")
    if args.jvm_gb < 1:
        fail("--jvm-gb must be positive")
    info = prepare(root, args.scope, dep_dir)
    if info["status"] != "PREPARED_NOT_REASONED":
        base.log("BLOCKED: missing/ambiguous recursive owl:imports. Report: " + str(root / OUT / "import_dependency_report.json"))
        return 2
    base.log(f"PREPARED {args.scope}: all four folds; no named subclass leakage detected")
    if args.prepare_only:
        return 0
    outcome = reason(root, info, args.jvm_gb)
    base.log("Result: " + outcome["status"])
    return 0 if outcome["status"].startswith("PASS_HERMIT") else 2

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("[v9 OWL audit v2] FAILED: " + str(exc), file=sys.stderr, flush=True)
        sys.exit(2)
