
"""Project 5, Step 4: deterministic hierarchy-aware train/validation split.

Input (only initial project data file): src/module.ttl
Outputs:
    src/train.ttl
    src/valid.ttl
    reports/split_summary.json

Run from any directory:
    python scripts/split_train_valid.py

Dependencies: rdflib (Python standard library otherwise).

Policy:
  * Approximately 80% / 20% split of directly asserted named subclass axioms.
  * The training ontology keeps all non-held-out source triples.
  * Both outputs preserve all class and property declaration triples by identity.
  * All endpoints of held-out edges must appear in at least one training edge.
  * No held-out edge may be reachable through other local named training edges.
  * Seek diverse validation parent classes across deterministic randomized attempts.
  * Never use an existing split or report as input.

Checks do NOT establish no entailment from OWL imports or complex axioms.
"""

import argparse
import json
import random
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "module.ttl"
TRAIN = ROOT / "src" / "train.ttl"
VALID = ROOT / "src" / "valid.ttl"
SUMMARY = ROOT / "reports" / "split_summary.json"

DECLARATION_KINDS = {
    OWL.Class,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
    OWL.NamedIndividual,
}


def named_edges(graph):
    return {
        (str(child), str(parent))
        for child, parent in graph.subject_objects(RDFS.subClassOf)
        if isinstance(child, URIRef)
        and isinstance(parent, URIRef)
        and child != parent
    }


def declaration_triples(graph):
    return {
        (s, p, o)
        for s, p, o in graph
        if p == RDF.type and o in DECLARATION_KINDS
    }


def endpoint_degrees(edges):
    degrees = Counter()

    for child, parent in edges:
        degrees[child] += 1
        degrees[parent] += 1

    return degrees


def reachable(edges, child, target):
    parents = defaultdict(set)

    for a, b in edges:
        parents[a].add(b)

    stack = list(parents.get(child, ()))
    visited = set()

    while stack:
        current = stack.pop()

        if current == target:
            return True

        if current not in visited:
            visited.add(current)
            stack.extend(
                parents.get(current, set()) - visited
            )

    return False


def validate_partition(
    source_edges,
    train_edges,
    valid_edges
):
    if train_edges & valid_edges:
        raise RuntimeError(
            "Training and validation share subclass edges"
        )

    if (train_edges | valid_edges) != source_edges:
        raise RuntimeError(
            "Training and validation do not partition the source"
        )

    degrees = endpoint_degrees(train_edges)

    all_endpoints = {
        node
        for edge in source_edges
        for node in edge
    }

    missing = all_endpoints - set(degrees)

    if missing:
        raise RuntimeError(
            f"Training lost {len(missing)} original subclass endpoints"
        )

    leaking = [
        edge
        for edge in sorted(valid_edges)
        if reachable(train_edges, edge[0], edge[1])
    ]

    if leaking:
        raise RuntimeError(
            f"{len(leaking)} held-out edges are reachable "
            "through local training edges"
        )

    return len(all_endpoints)


def diversity_score(held_out):
    counts = Counter(
        parent
        for _, parent in held_out
    )

    return (
        len(counts),
        -sum(
            count * count
            for count in counts.values()
        ),
    )


def one_attempt(source_edges, target, seed):
    rng = random.Random(seed)

    remaining = set(source_edges)
    held_out = set()
    degrees = endpoint_degrees(remaining)

    parent_totals = Counter(
        parent
        for _, parent in source_edges
    )

    selected_parents = Counter()

    for _ in range(target):
        eligible = []

        for child, parent in sorted(remaining):
            # Ensure validation endpoints remain visible in training.
            if (
                degrees[child] <= 1
                or degrees[parent] <= 1
            ):
                continue

            reduced = remaining - {(child, parent)}

            # Subsequent edge removal cannot create new paths.
            if reachable(reduced, child, parent):
                continue

            eligible.append((child, parent))

        if not eligible:
            break

        # Favor different parent classes while preserving constraints.
        def priority(edge):
            child, parent = edge
            represented = selected_parents[parent]

            return (
                -represented,
                -(represented / parent_totals[parent]),
                min(degrees[child], degrees[parent]),
                rng.random(),
            )

        chosen = max(eligible, key=priority)

        remaining.remove(chosen)
        held_out.add(chosen)

        degrees[chosen[0]] -= 1
        degrees[chosen[1]] -= 1

        selected_parents[chosen[1]] += 1

    return remaining, held_out


def choose_split(source_edges, target, seed, restarts):
    best = None
    best_key = None

    for attempt in range(restarts):
        remaining, held_out = one_attempt(
            source_edges,
            target,
            seed + attempt * 7919,
        )

        key = (
            len(held_out),
            *diversity_score(held_out),
        )

        if best_key is None or key > best_key:
            best = (remaining, held_out)
            best_key = key

        if (
            key[0] == target
            and key[1] == len({
                parent
                for _, parent in source_edges
            })
        ):
            break

    return best


def new_graph_like(source):
    graph = Graph()

    for prefix, namespace in source.namespaces():
        graph.bind(
            prefix,
            namespace,
            override=True,
        )

    return graph


def atomic_turtle(graph, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    try:
        graph.serialize(
            destination=str(tmp),
            format="turtle",
        )

        tmp.replace(path)

    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json(data, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    try:
        tmp.write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="utf-8",
        )

        tmp.replace(path)

    finally:
        if tmp.exists():
            tmp.unlink()


def run():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--restarts",
        type=int,
        default=250,
    )

    args = parser.parse_args()

    if not (
        0.0 < args.validation_fraction < 0.5
    ):
        parser.error(
            "Validation fraction must be between zero and 0.5"
        )

    if args.restarts < 1:
        parser.error(
            "Restarts must be positive"
        )

    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Initial ontology not found: {SOURCE}"
        )

    source = Graph()
    source.parse(
        str(SOURCE),
        format="turtle",
    )

    original_edges = named_edges(source)
    original_declarations = declaration_triples(source)

    if len(original_edges) < 2:
        raise ValueError(
            "Not enough named subclass edges to split"
        )

    if not original_declarations:
        raise ValueError(
            "No class or property declarations in module.ttl"
        )

    desired = max(
        1,
        round(
            len(original_edges)
            * args.validation_fraction
        ),
    )

    desired = min(
        desired,
        len(original_edges) - 1,
    )

    train_edges, valid_edges = choose_split(
        original_edges,
        desired,
        args.seed,
        args.restarts,
    )

    if not valid_edges:
        raise RuntimeError(
            "No leak-free validation edge can preserve "
            "training endpoint coverage"
        )

    endpoint_count = validate_partition(
        original_edges,
        train_edges,
        valid_edges,
    )

    train = new_graph_like(source)
    train += source

    valid = new_graph_like(source)

    for triple in original_declarations:
        valid.add(triple)

    for child, parent in valid_edges:
        triple = (
            URIRef(child),
            RDFS.subClassOf,
            URIRef(parent),
        )

        train.remove(triple)
        valid.add(triple)

    if (
        named_edges(train) != train_edges
        or named_edges(valid) != valid_edges
    ):
        raise RuntimeError(
            "Output graphs contain unexpected subclass edges"
        )

    if not original_declarations.issubset(
        declaration_triples(train)
    ):
        raise RuntimeError(
            "Training lost original declarations"
        )

    if not original_declarations.issubset(
        declaration_triples(valid)
    ):
        raise RuntimeError(
            "Validation lost original declarations"
        )

    if len(train) != len(source) - len(valid_edges):
        raise RuntimeError(
            "Training graph did not preserve other source triples"
        )

    validate_partition(
        original_edges,
        named_edges(train),
        named_edges(valid),
    )

    parent_counts = Counter(
        parent
        for _, parent in valid_edges
    )

    all_parents = {
        parent
        for _, parent in original_edges
    }

    achieved_ratio = (
        len(valid_edges) / len(original_edges)
    )

    summary = {
        "status": (
            "PASS"
            if len(valid_edges) == desired
            else "PASS_APPROXIMATE"
        ),
        "source_file": str(SOURCE),
        "train_file": str(TRAIN),
        "validation_file": str(VALID),
        "source_triples": len(source),
        "training_triples": len(train),
        "validation_triples": len(valid),
        "source_named_subclass_axioms": len(original_edges),
        "training_named_subclass_axioms": len(train_edges),
        "validation_named_subclass_axioms": len(valid_edges),
        "target_validation_axioms": desired,
        "target_validation_fraction": args.validation_fraction,
        "achieved_validation_fraction": achieved_ratio,
        "original_declaration_count": len(original_declarations),
        "training_declarations_preserved": True,
        "validation_declarations_preserved": True,
        "source_subclass_endpoints": endpoint_count,
        "all_source_subclass_endpoints_in_training": True,
        "local_direct_or_transitive_leaks": 0,
        "distinct_source_superclasses": len(all_parents),
        "distinct_validation_superclasses": len(parent_counts),
        "validation_superclass_distribution": dict(
            sorted(parent_counts.items())
        ),
        "seed": args.seed,
        "restarts": args.restarts,
        "limitations": [
            "Local graph reachability is not a complete OWL entailment check.",
            "Imports and complex axioms are not proven leak-free.",
            "Validation contains declarations and withheld named subclass axioms.",
            "Other source logical axioms remain in training only.",
            "Parent balancing is constrained by endpoint coverage.",
        ],
    }

    atomic_turtle(train, TRAIN)
    atomic_turtle(valid, VALID)
    atomic_json(summary, SUMMARY)

    print("=" * 67)
    print(
        "STEP 4: REPRODUCIBLE HIERARCHY-AWARE "
        "TRAIN/VALID SPLIT"
    )
    print("=" * 67)

    print(
        "Original named subclass pairs:",
        len(original_edges)
    )

    print(
        "Training named subclass pairs:",
        len(train_edges)
    )

    print(
        "Validation named subclass pairs:",
        len(valid_edges)
    )

    print(
        "Validation fraction:",
        f"{achieved_ratio:.1%}"
    )

    print(
        "Target fraction:",
        f"{args.validation_fraction:.1%}"
    )

    print(
        "Declarations preserved in both outputs:",
        len(original_declarations)
    )

    print(
        "Source endpoints represented in training:",
        endpoint_count
    )

    print(
        "Distinct validation superclasses:",
        len(parent_counts)
    )

    print(
        "Local subclass reachability leakage: 0"
    )

    print("Train file:", TRAIN)
    print("Validation file:", VALID)
    print("Summary:", SUMMARY)

    if len(valid_edges) < desired:
        print(
            "NOTE: 80/20 target constrained by "
            "endpoint coverage and leakage checks."
        )

    print(
        "STEP 4 STATUS:",
        summary["status"]
    )


if __name__ == "__main__":
    try:
        run()

    except Exception as exc:
        print(
            f"STEP 4 STATUS: FAIL - "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        traceback.print_exc()
        sys.exit(1)
