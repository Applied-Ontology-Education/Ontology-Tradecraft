#!/usr/bin/env python3
"""Project 6: read-only feasibility audit for eight-class rail-only cosine evaluation.

Works with the completed Stage 1 ontology; NO retraining, external API, or new
native packages. Writes ONLY reports/rail_holdout_feasibility.json.

    python -u scripts/diagnose_rail_holdouts.py --project-root .

Production Step 5A currently trains on named subclass edges and owl:Class
assertions. This audit deliberately evaluates support in that SAME projection,
not hypothetical property/restriction training that does not yet exist.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL, SKOS

VERSION = "project6-rail-holdout-feasibility-v1"
NS = "https://example.org/ontologies/intel-ontology#"
EXPECTED = {
    "train", "passenger_train", "railway_station", "rail_journey",
    "rail_journey_plan", "rail_ticket", "rail_reservation_information",
    "rail_passenger_role",
}
PASSENGER = "https://www.commoncoreontologies.org/AgentOntology/passenger"
IMPORT_EXTS = {".ttl", ".owl", ".rdf", ".xml", ".nt", ".n3", ".trig"}


def hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def parse(path: Path) -> Graph:
    """Parse only local files; never dereference owl:imports."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    formats = {".ttl": ("turtle",), ".owl": ("xml", "turtle"),
               ".rdf": ("xml", "turtle"), ".xml": ("xml",),
               ".nt": ("nt",), ".n3": ("n3",), ".trig": ("trig",)}
    errors = []
    for fmt in formats.get(path.suffix.lower(), ("turtle", "xml")):
        try:
            return Graph().parse(location=str(path), format=fmt)
        except Exception as exc:
            errors.append(f"{fmt}: {exc}")
    raise ValueError(f"Could not parse {path}: {'; '.join(errors)}")


def named_edges(graph: Graph) -> set[tuple[str, str]]:
    return {(str(s), str(o)) for s, o in graph.subject_objects(RDFS.subClassOf)
            if isinstance(s, URIRef) and isinstance(o, URIRef) and s != o}


def reaches(edges: set[tuple[str, str]], src: str, dest: str) -> bool:
    by_child: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        by_child[a].add(b)
    queue = [src]
    seen = {src}
    while queue:
        node = queue.pop()
        if node == dest:
            return True
        for parent in by_child[node]:
            if parent not in seen:
                seen.add(parent)
                queue.append(parent)
    return False


def imported_context(root: Path) -> tuple[Graph, list[str], list[str]]:
    graph = Graph()
    loaded, warnings = [], []
    d = root / "imports"
    if not d.is_dir():
        return graph, loaded, ["imports/ folder absent: source/import ancestry checks are limited."]
    for path in sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMPORT_EXTS):
        relative = path.relative_to(root).as_posix()
        # Never silently skip a malformed existing import: it affects leakage checking.
        imported = parse(path)
        for triple in imported:
            graph.add(triple)
        loaded.append(relative)
    if not loaded:
        warnings.append("No import graphs loaded; verify train-only ontology context on the project machine.")
    return graph, loaded, warnings


def analyze(root: Path, *, source: Path | None = None, augmented: Path | None = None,
            load_imports: bool = True, seed: int = 42) -> dict:
    root = root.resolve()
    source = (source or root / "src/module.ttl").resolve()
    augmented = (augmented or root / "generated/module_augmented.ttl").resolve()
    source_before, augmented_before = hash_file(source), hash_file(augmented)
    original, final = parse(source), parse(augmented)
    if len(original) != 421 or len(final) != 484:
        raise ValueError(f"Unexpected Stage 1 graph sizes: source={len(original)}, augmented={len(final)}; expected 421/484")
    prefix_preserved = augmented.read_bytes().startswith(source.read_bytes())
    if not prefix_preserved:
        raise ValueError("Augmented output does not preserve the original source as a byte prefix")

    source_classes = {str(n) for n in original.subjects(RDF.type, OWL.Class) if isinstance(n, URIRef)}
    new_classes = {str(n) for n in final.subjects(RDF.type, OWL.Class) if isinstance(n, URIRef)} - source_classes
    expected_iris = {NS + name for name in EXPECTED}
    if new_classes != expected_iris:
        raise ValueError(f"Unexpected new rail class set: missing={sorted(expected_iris-new_classes)}, extra={sorted(new_classes-expected_iris)}")
    new_object = {str(x) for x in final.subjects(RDF.type, OWL.ObjectProperty)
                  if isinstance(x, URIRef) and (x, RDF.type, OWL.ObjectProperty) not in original}
    new_data = {str(x) for x in final.subjects(RDF.type, OWL.DatatypeProperty)
                if isinstance(x, URIRef) and (x, RDF.type, OWL.DatatypeProperty) not in original}
    if (len(new_object), len(new_data)) != (4, 2):
        raise ValueError("Unexpected property declaration counts: expected four object and two data properties")
    new_subjects = {URIRef(x) for x in new_classes | new_object | new_data}
    rail_triples = {t for t in final if t[0] in new_subjects}
    if len(rail_triples) != 63:
        raise ValueError(f"Rail class/property subject triples={len(rail_triples)}; expected 63")
    all_rail_edges = {(a, b) for a, b in named_edges(final) if a in new_classes}
    if len(all_rail_edges) != 9:
        raise ValueError(f"Expected nine new rail named subclass assertions, got {len(all_rail_edges)}")

    imports, loaded, warnings = (imported_context(root) if load_imports
                                 else (Graph(), [], ["Imports not supplied to this offline audit."]))
    base_edges = named_edges(original) | named_edges(imports)
    old_entities = (set(source_classes) |
                    {str(n) for n in imports.subjects(RDF.type, OWL.Class) if isinstance(n, URIRef)} |
                    {node for edge in base_edges for node in edge})
    # Choose one primary target per class; the role's passenger-specific link
    # is more informative than its broader direct BFO Role link.
    chosen: dict[str, tuple[str, str]] = {}
    for child in sorted(new_classes):
        parents = sorted(parent for c, parent in all_rail_edges if c == child)
        if not parents:
            raise ValueError(f"Rail class has no named superclass axiom: {child}")
        if child == NS + "rail_passenger_role":
            if PASSENGER not in parents:
                raise ValueError("Expected passenger-role specialization absent")
            parent = PASSENGER
        elif len(parents) == 1:
            parent = parents[0]
        else:
            raise ValueError(f"Ambiguous primary target for {child}: {parents}")
        chosen[child] = (child, parent)

    classes = sorted(new_classes)
    random.Random(seed).shuffle(classes)
    folds = [classes[i:i+2] for i in range(0, len(classes), 2)]
    if len(folds) != 4 or any(len(fold) != 2 for fold in folds):
        raise AssertionError("Four-fold split must contain exactly two held-out rail classes per fold")

    by_class = {}
    per_fold = []
    for fold_no, heldout_classes in enumerate(folds, 1):
        withheld = {chosen[child] for child in heldout_classes}
        train_edges = (base_edges | all_rail_edges) - withheld
        fold_out = {"fold": fold_no, "held_out_rail_classes": heldout_classes,
                    "held_out_pairs": [{"child": a, "parent": b} for a, b in sorted(withheld)]}
        fold_results = []
        for child in heldout_classes:
            pair = chosen[child]
            incoming = sorted((a, b) for a, b in train_edges if b == child)
            outgoing = sorted((a, b) for a, b in train_edges if a == child)
            connected_to_existing_parent = any(reaches(train_edges, child, old) for old in old_entities)
            leaked = reaches(train_edges, *pair)
            # Property domain/range assertions are visible in the ontology, but
            # NOT used by the current Project 6 named-subclass-only mOWL model.
            schema_mentions = sorted({str(s) + " " + str(p).rsplit("/", 1)[-1]
                                      for s, p, o in final.triples((None, None, URIRef(child)))
                                      if p in (RDFS.domain, RDFS.range)})
            labels = len(list(final.objects(URIRef(child), RDFS.label)))
            definitions = (len(list(final.objects(URIRef(child), SKOS.definition))) +
                           len(list(final.objects(URIRef(child), URIRef("http://purl.obolibrary.org/obo/IAO_0000115")))))
            if leaked:
                status = "INVALID_TRANSITIVE_LEAKAGE"
            elif not outgoing:
                status = "COLD_START_NO_RETAINED_PARENT" if not incoming else "INCOMING_ONLY_NO_PARENT_ANCHOR"
            elif not connected_to_existing_parent:
                status = "PARENT_EXISTS_BUT_NO_OLD_ONTOLOGY_ANCHOR"
            else:
                status = "RETAINED_TAXONOMY_PARENT_SUPPORT"
            row = {
                "class_iri": child,
                "class_label": child.rsplit("#", 1)[-1].replace("_", " "),
                "primary_heldout_parent": pair[1],
                "fold": fold_no,
                "train_named_parent_count": len(outgoing),
                "train_named_child_count": len(incoming),
                "train_taxonomy_incident_edge_count": len(outgoing) + len(incoming),
                "connects_to_existing_ontology_via_parent_chain": connected_to_existing_parent,
                "heldout_pair_already_reachable_from_training": leaked,
                "property_domain_range_mentions_NOT_IN_CURRENT_MOWL_LOSS": schema_mentions,
                "english_labels": labels,
                "definitions": definitions,
                "taxonomy_only_support_status": status,
            }
            by_class[child] = row
            fold_results.append(row)
        fold_out["test_examples"] = len(fold_results)
        fold_out["transitive_leakage_count"] = sum(x["heldout_pair_already_reachable_from_training"] for x in fold_results)
        fold_out["cold_or_weak_support_count"] = sum(x["taxonomy_only_support_status"] != "RETAINED_TAXONOMY_PARENT_SUPPORT" for x in fold_results)
        per_fold.append(fold_out)

    rows = [by_class[x] for x in sorted(by_class)]
    problematic = [x for x in rows if x["taxonomy_only_support_status"] != "RETAINED_TAXONOMY_PARENT_SUPPORT"]
    result = {
        "version": VERSION,
        "status": "DIAGNOSTIC_COMPLETE_NEEDS_RICHER_TRAINING_EVIDENCE" if problematic else "DIAGNOSTIC_COMPLETE_SPLIT_HAS_TAXONOMY_SUPPORT",
        "scope": "EIGHT_NEW_RAIL_CLASSES_ONLY",
        "files": {"source": str(source), "augmented": str(augmented)},
        "sha256": {"source": source_before, "augmented": augmented_before},
        "source_byte_prefix_preserved": prefix_preserved,
        "source_triples": len(original), "augmented_triples": len(final),
        "new_rail_subject_triples": len(rail_triples),
        "new_rail_classes": len(new_classes), "new_rail_object_properties": len(new_object),
        "new_rail_data_properties": len(new_data),
        "new_rail_subclass_relationships": len(all_rail_edges),
        "primary_target_pairs": len(chosen),
        "method": "Four folds with two primary rail superclass assertions withheld per fold; source/import subclass pairs are context only and NOT scored",
        "seed": seed,
        "folds": per_fold, "per_class": rows,
        "summary": {
            "folds": len(folds),
            "heldout_examples_per_fold": [len(x) for x in folds],
            "unique_rail_class_scores_if_trained": len(rows),
            "heldout_transitive_leakage_count": sum(x["heldout_pair_already_reachable_from_training"] for x in rows),
            "cold_or_weakly_supported_rail_classes": len(problematic),
            "supported_rail_classes": len(rows) - len(problematic),
        },
        "imports_loaded": loaded, "warnings": warnings,
        "interpretation": [
            "No embeddings were trained or scored by this diagnostic.",
            "The current Step 5A projects only named subclass pairs and class declarations.",
            "Labels, definitions, and property domain/range mentions are NOT trainable evidence in that projection.",
            "A class with no retained superclass has no established parent-class embedding anchor; a 0.70 cosine requirement is not supported by this split alone.",
            "A positive retained-parent status indicates topology support, not guaranteed cosine accuracy.",
        ],
        "files_unchanged": (source_before == hash_file(source) and augmented_before == hash_file(augmented)),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--source", type=Path, help="Optional location of original module.ttl")
    p.add_argument("--augmented", type=Path, help="Optional location of module_augmented.ttl")
    p.add_argument("--no-imports", action="store_true", help="Offline structural check without imports")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=Path, help="Report location (default PROJECT/reports/rail_holdout_feasibility.json)")
    args = p.parse_args(argv)
    try:
        result = analyze(args.project_root, source=args.source, augmented=args.augmented,
                         load_imports=not args.no_imports, seed=args.seed)
        output = args.report or args.project_root / "reports/rail_holdout_feasibility.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("RAIL HOLDOUT AUDIT — " + result["status"])
        print("Rail classes evaluated:", result["new_rail_classes"])
        print("Four-fold test counts:", result["summary"]["heldout_examples_per_fold"])
        print("Leaking heldouts:", result["summary"]["heldout_transitive_leakage_count"])
        print("Classes with insufficient taxonomy support:", result["summary"]["cold_or_weakly_supported_rail_classes"])
        print("REPORT:", output.resolve())
        return 0
    except Exception as exc:
        print("RAIL HOLDOUT AUDIT ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
