from pathlib import Path
from collections import defaultdict, deque
import argparse
import csv
import json

import numpy as np
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL


SCRIPT_VERSION = "validation-support-audit-v2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = SRC_DIR / "train.ttl"
VALID_FILE = SRC_DIR / "valid.ttl"

OUTPUT_CSV = REPORTS_DIR / "validation_support_audit.csv"
OUTPUT_JSON = REPORTS_DIR / "validation_support_audit.json"


DECLARATION_TYPES = {
    OWL.Class,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
    OWL.NamedIndividual,
    RDFS.Datatype,
}


LOGICAL_PREDICATES = {
    RDF.type,

    RDFS.subClassOf,
    RDFS.domain,
    RDFS.range,

    OWL.equivalentClass,
    OWL.disjointWith,
    OWL.complementOf,

    OWL.someValuesFrom,
    OWL.allValuesFrom,
    OWL.hasValue,
    OWL.hasSelf,

    OWL.intersectionOf,
    OWL.unionOf,
    OWL.oneOf,

    OWL.onClass,
    OWL.onDataRange,

    OWL.cardinality,
    OWL.minCardinality,
    OWL.maxCardinality,

    OWL.qualifiedCardinality,
    OWL.minQualifiedCardinality,
    OWL.maxQualifiedCardinality,

    RDF.first,
}


AUTO_SCORE_FILES = [
    REPORTS_DIR / "mowl_gentle_best_validation_pairs.csv",
    REPORTS_DIR / "mowl_local_validation_pairs.csv",
]


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=Path,
        default=TRAIN_FILE,
    )

    parser.add_argument(
        "--valid",
        type=Path,
        default=VALID_FILE,
    )

    parser.add_argument(
        "--scores-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--poor-threshold",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--minimum-support",
        type=int,
        default=1,
    )

    return parser.parse_args()


def load_graph(path):

    if not path.exists():

        raise FileNotFoundError(
            str(path)
        )

    graph = Graph()

    graph.parse(
        str(path),
        format="turtle",
    )

    return graph


def named_subclass_pairs(graph):

    pairs = set()

    for subclass, superclass in graph.subject_objects(
        RDFS.subClassOf
    ):

        if (
            isinstance(
                subclass,
                URIRef,
            )
            and
            isinstance(
                superclass,
                URIRef,
            )
            and
            subclass != superclass
        ):

            pairs.add(
                (
                    str(subclass),
                    str(superclass),
                )
            )

    return sorted(
        pairs
    )


def declaration_triple(
    triple,
):

    s, p, o = triple

    return (
        p == RDF.type
        and o in DECLARATION_TYPES
    )


def logical_support(
    graph,
    iri,
):

    node = URIRef(
        iri
    )

    count = 0

    for triple in graph:

        s, p, o = triple

        if (
            s != node
            and o != node
        ):
            continue

        if (
            s == node
            and declaration_triple(
                triple
            )
        ):
            continue

        if p not in LOGICAL_PREDICATES:
            continue

        count += 1

    return count


def hierarchy_support(
    graph,
    iri,
):

    node = URIRef(
        iri
    )

    outgoing = sum(
        1
        for _ in graph.objects(
            node,
            RDFS.subClassOf,
        )
    )

    incoming = sum(
        1
        for _ in graph.subjects(
            RDFS.subClassOf,
            node,
        )
    )

    return (
        outgoing,
        incoming,
    )


def declared(
    graph,
    iri,
):

    return (
        URIRef(iri),
        RDF.type,
        OWL.Class,
    ) in graph


def label_for(
    iri,
    *graphs,
):

    node = URIRef(
        iri
    )

    for graph in graphs:

        for label in graph.objects(
            node,
            RDFS.label,
        ):

            return str(
                label
            )

    if "#" in iri:

        return iri.rsplit(
            "#",
            1,
        )[-1]

    return iri.rstrip(
        "/"
    ).rsplit(
        "/",
        1,
    )[-1]


def transitive_closure(
    pairs,
):

    parents = defaultdict(set)

    nodes = set()

    for child, parent in pairs:

        parents[
            child
        ].add(
            parent
        )

        nodes.add(
            child
        )

        nodes.add(
            parent
        )

    closure = set()

    for child in nodes:

        queue = deque(
            parents.get(
                child,
                set(),
            )
        )

        seen = set()

        while queue:

            ancestor = queue.popleft()

            if ancestor in seen:
                continue

            seen.add(
                ancestor
            )

            if ancestor != child:

                closure.add(
                    (
                        child,
                        ancestor,
                    )
                )

            for parent in parents.get(
                ancestor,
                set(),
            ):

                if parent not in seen:

                    queue.append(
                        parent
                    )

    return closure


def load_score_file(path):

    result = {}

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        required = {
            "cosine",
            "subclass",
            "superclass",
        }

        if not required.issubset(
            set(
                reader.fieldnames
                or []
            )
        ):

            return {}

        for row in reader:

            result[
                (
                    row[
                        "subclass"
                    ].strip(),

                    row[
                        "superclass"
                    ].strip(),
                )
            ] = float(
                row[
                    "cosine"
                ]
            )

    return result


def select_scores(
    requested,
    valid_pairs,
):

    valid_set = set(
        valid_pairs
    )

    if requested is not None:

        if not requested.exists():

            raise FileNotFoundError(
                str(
                    requested
                )
            )

        scores = load_score_file(
            requested
        )

        return (
            requested,
            scores,
        )

    # Automatically use an old score file only if its pair
    # set exactly matches the current validation split.
    for candidate in AUTO_SCORE_FILES:

        if not candidate.exists():
            continue

        scores = load_score_file(
            candidate
        )

        if (
            set(
                scores.keys()
            )
            ==
            valid_set
        ):

            return (
                candidate,
                scores,
            )

    return (
        None,
        {},
    )


def main():

    args = parse_arguments()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_graph = load_graph(
        args.train
    )

    valid_graph = load_graph(
        args.valid
    )

    train_pairs = named_subclass_pairs(
        train_graph
    )

    valid_pairs = named_subclass_pairs(
        valid_graph
    )

    score_file, scores = select_scores(
        args.scores_file,
        valid_pairs,
    )

    direct_overlap = (
        set(
            train_pairs
        )
        &
        set(
            valid_pairs
        )
    )

    closure = transitive_closure(
        train_pairs
    )

    transitive_overlap = (
        closure
        &
        set(
            valid_pairs
        )
    )

    rows = []

    isolated = []

    below_minimum = []

    for subclass, superclass in valid_pairs:

        support = logical_support(
            train_graph,
            subclass,
        )

        outgoing, incoming = hierarchy_support(
            train_graph,
            subclass,
        )

        declaration_only = (
            declared(
                train_graph,
                subclass,
            )
            and support == 0
        )

        if support == 0:

            isolated.append(
                (
                    subclass,
                    superclass,
                )
            )

        if support < args.minimum_support:

            below_minimum.append(
                (
                    subclass,
                    superclass,
                )
            )

        cosine = scores.get(
            (
                subclass,
                superclass,
            )
        )

        rows.append(
            {
                "cosine":
                    cosine,

                "subclass":
                    subclass,

                "subclass_label":
                    label_for(
                        subclass,
                        train_graph,
                        valid_graph,
                    ),

                "superclass":
                    superclass,

                "superclass_label":
                    label_for(
                        superclass,
                        train_graph,
                        valid_graph,
                    ),

                "logical_support":
                    support,

                "outgoing_subclass_edges":
                    outgoing,

                "incoming_subclass_edges":
                    incoming,

                "declared":
                    declared(
                        train_graph,
                        subclass,
                    ),

                "declaration_only":
                    declaration_only,
            }
        )

    rows.sort(
        key=lambda row: (
            row[
                "logical_support"
            ],
            (
                row[
                    "cosine"
                ]
                if row[
                    "cosine"
                ]
                is not None
                else 999
            ),
        )
    )

    print(
        f"VALIDATION_SUPPORT_VERSION : "
        f"{SCRIPT_VERSION}"
    )

    print()

    print("=" * 76)

    print(
        "PROJECT 5 - SPLIT HEALTH / VALIDATION SUPPORT AUDIT"
    )

    print("=" * 76)

    print()

    print(
        f"Training named subclass axioms   : "
        f"{len(train_pairs)}"
    )

    print(
        f"Validation named subclass axioms : "
        f"{len(valid_pairs)}"
    )

    print()

    print(
        f"Direct overlap                   : "
        f"{len(direct_overlap)}"
    )

    print(
        f"Transitive validation leakage    : "
        f"{len(transitive_overlap)}"
    )

    print(
        f"Structurally isolated subclasses : "
        f"{len(isolated)}"
    )

    print(
        f"Below minimum support ({args.minimum_support})       : "
        f"{len(below_minimum)}"
    )

    print()

    if score_file is None:

        print(
            "Cosine scores                    : "
            "NOT LOADED (structural preflight mode)"
        )

    else:

        print(
            f"Cosine scores                    : "
            f"{score_file}"
        )

    print()

    print(
        "Support  Out  In   Cosine    "
        "Subclass -> Superclass"
    )

    print(
        "-" * 120
    )

    for row in rows:

        cosine_text = (
            f"{row['cosine']:+.4f}"
            if row[
                "cosine"
            ]
            is not None
            else "   N/A "
        )

        print(
            f"{row['logical_support']:>7} "
            f"{row['outgoing_subclass_edges']:>4} "
            f"{row['incoming_subclass_edges']:>3} "
            f"{cosine_text:>9}   "
            f"{row['subclass_label']} "
            f"-> "
            f"{row['superclass_label']}"
        )

    healthy = (
        len(
            direct_overlap
        ) == 0

        and
        len(
            transitive_overlap
        ) == 0

        and
        len(
            isolated
        ) == 0

        and
        len(
            below_minimum
        ) == 0
    )

    print()

    print("=" * 76)

    if healthy:

        print(
            "SPLIT HEALTH: PASS"
        )

    else:

        print(
            "SPLIT HEALTH: FAIL"
        )

    print("=" * 76)

    report = {
        "script_version":
            SCRIPT_VERSION,

        "train_pairs":
            len(
                train_pairs
            ),

        "validation_pairs":
            len(
                valid_pairs
            ),

        "direct_overlap":
            len(
                direct_overlap
            ),

        "transitive_overlap":
            len(
                transitive_overlap
            ),

        "isolated_validation_subclasses":
            len(
                isolated
            ),

        "below_minimum_support":
            len(
                below_minimum
            ),

        "minimum_support":
            args.minimum_support,

        "split_health_pass":
            healthy,

        "score_file":
            (
                str(
                    score_file
                )
                if score_file
                else None
            ),

        "rows":
            rows,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2,
        )

    fields = [
        "cosine",
        "logical_support",
        "outgoing_subclass_edges",
        "incoming_subclass_edges",
        "declared",
        "declaration_only",
        "subclass_label",
        "subclass",
        "superclass_label",
        "superclass",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    field:
                        row.get(
                            field,
                            ""
                        )
                    for field
                    in fields
                }
            )

    print()

    print(
        f"Report:\n"
        f"  {OUTPUT_JSON}"
    )

    print()

    if not healthy:

        raise SystemExit(
            2
        )


if __name__ == "__main__":

    main()