from pathlib import Path
from collections import defaultdict, deque
import argparse
import csv
import json

import numpy as np
from rdflib import Graph, URIRef
from rdflib.namespace import RDFS

import mowl


# ============================================================
# VERSION
# ============================================================

SCRIPT_VERSION = "mowl-training-coverage-audit-v2"


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODULE_FILE = SRC_DIR / "module.ttl"
TRAIN_FILE = SRC_DIR / "train.ttl"
VALID_FILE = SRC_DIR / "valid.ttl"

OUTPUT_CSV = (
    REPORTS_DIR
    / "mowl_training_coverage_audit.csv"
)

OUTPUT_JSON = (
    REPORTS_DIR
    / "mowl_training_coverage_audit.json"
)

OUTPUT_TXT = (
    REPORTS_DIR
    / "mowl_training_coverage_summary.txt"
)


# ============================================================
# MOWL EL NORMAL-FORM CLASS COLUMNS
#
# GCI0:
#     C ⊑ D
#
# GCI1:
#     C1 ⊓ C2 ⊑ D
#
# GCI2:
#     C ⊑ ∃R.D
#
# GCI3:
#     ∃R.C ⊑ D
#
# Object-property columns are intentionally excluded.
# ============================================================

GCI_CLASS_COLUMNS = {

    "gci0": [
        (0, "lhs_class"),
        (1, "rhs_class"),
    ],

    "gci1": [
        (0, "lhs_conjunct_1"),
        (1, "lhs_conjunct_2"),
        (2, "rhs_class"),
    ],

    "gci2": [
        (0, "lhs_class"),
        (2, "rhs_existential_filler"),
    ],

    "gci3": [
        (1, "lhs_existential_filler"),
        (2, "rhs_class"),
    ],

    "gci0_bot": [
        (0, "lhs_class"),
    ],

    "gci1_bot": [
        (0, "lhs_conjunct_1"),
        (1, "lhs_conjunct_2"),
    ],

    "gci3_bot": [
        (1, "lhs_existential_filler"),
    ],
}


LHS_ROLES = {
    "lhs_class",
    "lhs_conjunct_1",
    "lhs_conjunct_2",
    "lhs_existential_filler",
}


RHS_ROLES = {
    "rhs_class",
    "rhs_existential_filler",
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Inspect actual normalized mOWL ELEmbeddings "
            "training support for Project 5 validation classes."
        )
    )

    parser.add_argument(
        "--module",
        type=Path,
        default=MODULE_FILE,
    )

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
        help=(
            "Optional validation-score CSV containing cosine, "
            "subclass, and superclass. If omitted, the script "
            "runs in pre-training structural diagnostic mode."
        ),
    )

    parser.add_argument(
        "--jvm-memory",
        default="2g",
    )

    parser.add_argument(
        "--embed-dim",
        type=int,
        default=50,
    )

    return parser.parse_args()


# ============================================================
# RDF HELPERS
# ============================================================

def load_graph(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    graph = Graph()

    graph.parse(
        str(path),
        format="turtle",
    )

    return graph


def load_named_subclass_pairs(graph):

    pairs = set()

    for subclass, superclass in graph.subject_objects(
        RDFS.subClassOf
    ):

        if not isinstance(
            subclass,
            URIRef,
        ):
            continue

        if not isinstance(
            superclass,
            URIRef,
        ):
            continue

        if subclass == superclass:
            continue

        pairs.add(
            (
                str(subclass),
                str(superclass),
            )
        )

    return sorted(
        pairs
    )


def get_label(
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


# ============================================================
# OPTIONAL SCORE FILE
# ============================================================

def load_scores(path):

    if path is None:

        return {}

    if not path.exists():

        raise FileNotFoundError(
            f"Score file not found: {path}"
        )

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

        fields = set(
            reader.fieldnames
            or []
        )

        if not required.issubset(
            fields
        ):

            raise ValueError(
                "Validation-score CSV must contain "
                "cosine, subclass, and superclass columns."
            )

        for row in reader:

            key = (
                row["subclass"].strip(),
                row["superclass"].strip(),
            )

            result[
                key
            ] = float(
                row["cosine"]
            )

    return result


# ============================================================
# TRANSITIVE CLOSURE
# ============================================================

def build_parent_map(
    pairs,
):

    parents = defaultdict(
        set
    )

    for child, parent in pairs:

        parents[
            child
        ].add(
            parent
        )

    return parents


def transitive_closure(
    pairs,
):

    parents = build_parent_map(
        pairs
    )

    nodes = set(
        parents.keys()
    )

    for values in parents.values():

        nodes.update(
            values
        )

    closure = set()

    for child in nodes:

        queue = deque(
            parents.get(
                child,
                set(),
            )
        )

        visited = set()

        while queue:

            ancestor = queue.popleft()

            if ancestor in visited:
                continue

            visited.add(
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

                if parent not in visited:

                    queue.append(
                        parent
                    )

    return closure


# ============================================================
# VALIDATION-EDGE CLOSURE IMPACT
# ============================================================

def closure_impact_for_edge(
    edge,
    module_pairs,
    train_pairs,
    module_closure,
    train_closure,
):

    full_pair_set = set(
        module_pairs
    )

    train_pair_set = set(
        train_pairs
    )

    # --------------------------------------------------------
    # Remove only this validation edge from the full module.
    # --------------------------------------------------------

    module_without_edge = (
        full_pair_set
        -
        {
            edge
        }
    )

    closure_without_edge = (
        transitive_closure(
            module_without_edge
        )
    )

    individually_lost = (
        module_closure
        -
        closure_without_edge
    )

    # --------------------------------------------------------
    # Put this validation edge back into the current training
    # hierarchy to see how much closure it restores.
    # --------------------------------------------------------

    train_plus_edge = (
        train_pair_set
        |
        {
            edge
        }
    )

    closure_plus_edge = (
        transitive_closure(
            train_plus_edge
        )
    )

    marginal_restored = (
        closure_plus_edge
        -
        train_closure
    )

    return {

        "edge_still_entailed_without_it":
            (
                edge
                in closure_without_edge
            ),

        "individual_closure_pairs_lost":
            len(
                individually_lost
            ),

        "marginal_closure_pairs_restored":
            len(
                marginal_restored
            ),
    }


# ============================================================
# CLASS INDEX
# ============================================================

def normalize_class_index(
    class_index_dict,
):

    return {
        str(iri):
            int(index)

        for iri, index
        in class_index_dict.items()
    }


# ============================================================
# DATASET -> NUMPY
# ============================================================

def dataset_to_numpy(
    dataset,
):

    if len(
        dataset
    ) == 0:

        return np.empty(
            (
                0,
                0,
            ),
            dtype=np.int64,
        )

    values = dataset[:]

    if hasattr(
        values,
        "detach",
    ):

        values = (
            values
            .detach()
            .cpu()
            .numpy()
        )

    else:

        values = np.asarray(
            values
        )

    values = np.asarray(
        values
    )

    if values.ndim == 0:

        values = values.reshape(
            1,
            1,
        )

    elif values.ndim == 1:

        values = values.reshape(
            1,
            -1,
        )

    return values


# ============================================================
# PREPARE NORMALIZED TRAINING DATASETS
# ============================================================

def prepare_training_arrays(
    model,
):

    arrays = {}

    summary = {}

    for (
        gci_name,
        dataset,
    ) in model.training_datasets.items():

        array = dataset_to_numpy(
            dataset
        )

        arrays[
            gci_name
        ] = array

        summary[
            gci_name
        ] = {

            "rows":
                int(
                    len(
                        dataset
                    )
                ),

            "columns":
                (
                    int(
                        array.shape[1]
                    )
                    if (
                        array.ndim == 2
                        and
                        array.size > 0
                    )
                    else 0
                ),
        }

    return (
        arrays,
        summary,
    )


# ============================================================
# ACTUAL MOWL CLASS COVERAGE
# ============================================================

def class_training_coverage(
    iri,
    class_index,
    arrays,
):

    if iri not in class_index:

        return {

            "indexed":
                False,

            "class_id":
                None,

            "total_occurrences":
                0,

            "unique_training_rows":
                0,

            "lhs_occurrences":
                0,

            "rhs_occurrences":
                0,

            "gci_types_present":
                0,
        }

    class_id = class_index[
        iri
    ]

    total_occurrences = 0
    unique_rows_total = 0

    lhs_occurrences = 0
    rhs_occurrences = 0

    gci_types_present = 0

    per_gci_occurrences = {}
    per_gci_rows = {}

    per_role = defaultdict(
        int
    )

    for (
        gci_name,
        array,
    ) in arrays.items():

        schema = GCI_CLASS_COLUMNS.get(
            gci_name
        )

        if schema is None:
            continue

        if (
            array.ndim != 2
            or
            array.shape[0] == 0
        ):

            per_gci_occurrences[
                gci_name
            ] = 0

            per_gci_rows[
                gci_name
            ] = 0

            continue

        row_mask = np.zeros(
            array.shape[0],
            dtype=bool,
        )

        gci_occurrences = 0

        for column, role in schema:

            if column >= array.shape[1]:
                continue

            values = array[
                :,
                column
            ]

            matches = (
                values
                ==
                class_id
            )

            count = int(
                np.sum(
                    matches
                )
            )

            gci_occurrences += count

            row_mask |= matches

            per_role[
                f"{gci_name}:{role}"
            ] += count

            if role in LHS_ROLES:

                lhs_occurrences += count

            if role in RHS_ROLES:

                rhs_occurrences += count

        gci_rows = int(
            np.sum(
                row_mask
            )
        )

        per_gci_occurrences[
            gci_name
        ] = gci_occurrences

        per_gci_rows[
            gci_name
        ] = gci_rows

        total_occurrences += (
            gci_occurrences
        )

        unique_rows_total += (
            gci_rows
        )

        if gci_rows > 0:

            gci_types_present += 1

    result = {

        "indexed":
            True,

        "class_id":
            class_id,

        "total_occurrences":
            total_occurrences,

        "unique_training_rows":
            unique_rows_total,

        "lhs_occurrences":
            lhs_occurrences,

        "rhs_occurrences":
            rhs_occurrences,

        "gci_types_present":
            gci_types_present,
    }

    for gci_name in [
        "gci0",
        "gci1",
        "gci2",
        "gci3",
        "gci0_bot",
        "gci1_bot",
        "gci3_bot",
    ]:

        result[
            f"{gci_name}_occurrences"
        ] = per_gci_occurrences.get(
            gci_name,
            0,
        )

        result[
            f"{gci_name}_rows"
        ] = per_gci_rows.get(
            gci_name,
            0,
        )

    result[
        "role_counts"
    ] = dict(
        per_role
    )

    return result


# ============================================================
# CORRELATION
# ============================================================

def pearson(
    x,
    y,
):

    if len(
        x
    ) < 2:

        return None

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if (
        np.std(
            x
        ) == 0
        or
        np.std(
            y
        ) == 0
    ):

        return None

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def rank_values(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    order = np.argsort(
        values
    )

    ranks = np.empty(
        len(
            values
        ),
        dtype=float,
    )

    i = 0

    while i < len(
        values
    ):

        j = i

        while (
            j + 1
            <
            len(
                values
            )
            and
            values[
                order[j + 1]
            ]
            ==
            values[
                order[i]
            ]
        ):

            j += 1

        average_rank = (
            (
                i
                +
                j
            )
            /
            2.0
            +
            1.0
        )

        for position in range(
            i,
            j + 1,
        ):

            ranks[
                order[position]
            ] = average_rank

        i = j + 1

    return ranks


def spearman(
    x,
    y,
):

    if len(
        x
    ) < 2:

        return None

    return pearson(
        rank_values(
            x
        ),
        rank_values(
            y
        ),
    )


# ============================================================
# CSV
# ============================================================

def save_csv(
    rows,
):

    fields = [

        "cosine",

        "subclass_label",
        "subclass",

        "superclass_label",
        "superclass",

        "sub_mowl_total_occurrences",
        "sub_mowl_unique_training_rows",

        "sub_mowl_lhs_occurrences",
        "sub_mowl_rhs_occurrences",

        "sub_mowl_gci_types_present",

        "sub_gci0_rows",
        "sub_gci1_rows",
        "sub_gci2_rows",
        "sub_gci3_rows",

        "sub_gci0_bot_rows",
        "sub_gci1_bot_rows",
        "sub_gci3_bot_rows",

        "super_mowl_total_occurrences",
        "super_mowl_unique_training_rows",

        "super_mowl_lhs_occurrences",
        "super_mowl_rhs_occurrences",

        "super_mowl_gci_types_present",

        "edge_still_entailed_without_it",

        "individual_closure_pairs_lost",
        "marginal_closure_pairs_restored",

        "zero_subclass_mowl_signal",
        "zero_superclass_mowl_signal",
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

                    for field in fields
                }
            )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD RDF
    # ========================================================

    module_graph = load_graph(
        args.module
    )

    train_graph = load_graph(
        args.train
    )

    valid_graph = load_graph(
        args.valid
    )

    module_pairs = (
        load_named_subclass_pairs(
            module_graph
        )
    )

    train_pairs = (
        load_named_subclass_pairs(
            train_graph
        )
    )

    valid_pairs = (
        load_named_subclass_pairs(
            valid_graph
        )
    )

    # ========================================================
    # OPTIONAL SCORES
    # ========================================================

    if args.scores_file is not None:

        scores = load_scores(
            args.scores_file
        )

    else:

        scores = {}

    valid_set = set(
        valid_pairs
    )

    score_set = set(
        scores.keys()
    )

    # ========================================================
    # HEADER
    # ========================================================

    print(
        f"MOWL_COVERAGE_VERSION : "
        f"{SCRIPT_VERSION}"
    )

    print()

    print("=" * 78)

    print(
        "PROJECT 5 - ACTUAL MOWL TRAINING COVERAGE AUDIT"
    )

    print("=" * 78)

    print()

    print(
        f"Module                 : "
        f"{args.module}"
    )

    print(
        f"Train                  : "
        f"{args.train}"
    )

    print(
        f"Valid                  : "
        f"{args.valid}"
    )

    if args.scores_file is None:

        print(
            "Scores                 : NOT LOADED "
            "(pre-training diagnostic mode)"
        )

    else:

        print(
            f"Scores                 : "
            f"{args.scores_file}"
        )

    print()

    print(
        f"Module subclass pairs  : "
        f"{len(module_pairs)}"
    )

    print(
        f"Train subclass pairs   : "
        f"{len(train_pairs)}"
    )

    print(
        f"Valid subclass pairs   : "
        f"{len(valid_pairs)}"
    )

    if scores:

        print(
            f"Score pairs            : "
            f"{len(scores)}"
        )

        if score_set == valid_set:

            print(
                "Score/validation match : PASS"
            )

        else:

            print(
                "Score/validation match : FAIL"
            )

            print()

            print(
                f"Missing score pairs    : "
                f"{len(valid_set - score_set)}"
            )

            print(
                f"Stale score pairs      : "
                f"{len(score_set - valid_set)}"
            )

            raise RuntimeError(
                "Score CSV does not correspond exactly "
                "to the current validation split."
            )

    else:

        print(
            "Score pairs            : 0"
        )

        print(
            "Score/validation match : SKIPPED "
            "(no score file supplied)"
        )

    print()

    # ========================================================
    # CLOSURE
    # ========================================================

    module_closure = (
        transitive_closure(
            module_pairs
        )
    )

    train_closure = (
        transitive_closure(
            train_pairs
        )
    )

    total_closure_lost = (
        module_closure
        -
        train_closure
    )

    retention = (
        len(
            train_closure
        )
        /
        len(
            module_closure
        )
        if module_closure
        else 0.0
    )

    print(
        "Named-subclass closure:"
    )

    print(
        f"  Full module closure  : "
        f"{len(module_closure)}"
    )

    print(
        f"  Current train closure: "
        f"{len(train_closure)}"
    )

    print(
        f"  Closure pairs lost   : "
        f"{len(total_closure_lost)}"
    )

    print(
        f"  Closure retention    : "
        f"{retention:.1%}"
    )

    print()

    # ========================================================
    # JVM
    # ========================================================

    print(
        "Initializing mOWL JVM..."
    )

    mowl.init_jvm(
        args.jvm_memory
    )

    from mowl.datasets import (
        PathDataset,
        Dataset,
    )

    from mowl.models import (
        ELEmbeddings,
    )

    from org.semanticweb.owlapi.apibinding import (
        OWLManager,
    )

    from java.util import (
        HashSet,
    )

    print(
        "JVM initialized."
    )

    print()

    # ========================================================
    # RECREATE SAME ONTOLOGY USED BY TRAIN_MOWL V4
    #
    # Root logical axioms only.
    # Imported vocabulary retained as declarations.
    # Imported logical axioms excluded.
    # ========================================================

    print(
        "Building v4-equivalent training ontology..."
    )

    path_dataset = PathDataset(
        str(
            args.train.resolve()
        )
    )

    root_ontology = (
        path_dataset.ontology
    )

    import_closure = list(
        root_ontology.getImportsClosure()
    )

    root_axioms = (
        root_ontology.getAxioms()
    )

    manager = (
        OWLManager
        .createOWLOntologyManager()
    )

    data_factory = (
        manager.getOWLDataFactory()
    )

    training_axioms = (
        HashSet()
    )

    training_axioms.addAll(
        root_axioms
    )

    broad_signature = (
        HashSet()
    )

    for ontology in import_closure:

        broad_signature.addAll(
            ontology.getSignature()
        )

    declarations_added = 0

    for entity in broad_signature:

        declaration = (
            data_factory
            .getOWLDeclarationAxiom(
                entity
            )
        )

        if not training_axioms.contains(
            declaration
        ):

            training_axioms.add(
                declaration
            )

            declarations_added += 1

    training_ontology = (
        manager.createOntology(
            training_axioms
        )
    )

    dataset = Dataset(
        training_ontology
    )

    print(
        f"Import closure          : "
        f"{len(import_closure)}"
    )

    print(
        f"Root logical axioms     : "
        f"{root_axioms.size()}"
    )

    print(
        f"Broad signature entities: "
        f"{broad_signature.size()}"
    )

    print(
        f"Declarations added      : "
        f"{declarations_added}"
    )

    print(
        f"Indexed classes         : "
        f"{len(dataset.classes.as_str)}"
    )

    print(
        f"Object properties       : "
        f"{len(dataset.object_properties.as_str)}"
    )

    print(
        "Imported logical axioms: NOT INCLUDED"
    )

    print()

    # ========================================================
    # CREATE MODEL
    #
    # NO TRAINING IS PERFORMED.
    # ========================================================

    model = ELEmbeddings(

        dataset,

        embed_dim=(
            args.embed_dim
        ),

        margin=0.1,

        reg_norm=1.0,

        learning_rate=0.001,

        batch_size=128,

        model_filepath=None,

        device="cpu",
    )

    class_index = normalize_class_index(
        model.class_index_dict
    )

    (
        training_arrays,
        dataset_summary,
    ) = prepare_training_arrays(
        model
    )

    # ========================================================
    # NORMALIZED DATASETS
    # ========================================================

    print("=" * 78)

    print(
        "ACTUAL MOWL NORMALIZED TRAINING DATASETS"
    )

    print("=" * 78)

    print()

    for gci_name in sorted(
        dataset_summary
    ):

        info = dataset_summary[
            gci_name
        ]

        print(
            f"{gci_name:<12} "
            f"rows={info['rows']:<5} "
            f"columns={info['columns']}"
        )

    print()

    unknown_gcis = [
        name
        for name in training_arrays
        if name not in GCI_CLASS_COLUMNS
    ]

    if unknown_gcis:

        print(
            "WARNING: unrecognized mOWL dataset types "
            "were ignored for class-role counts:"
        )

        for name in unknown_gcis:

            print(
                f"  {name}"
            )

        print()

    # ========================================================
    # CLASS COVERAGE CACHE
    # ========================================================

    coverage_cache = {}

    def coverage(iri):

        if iri not in coverage_cache:

            coverage_cache[
                iri
            ] = class_training_coverage(
                iri,
                class_index,
                training_arrays,
            )

        return coverage_cache[
            iri
        ]

    # ========================================================
    # VALIDATION PAIRS
    # ========================================================

    rows = []

    for edge in valid_pairs:

        subclass, superclass = edge

        sub = coverage(
            subclass
        )

        sup = coverage(
            superclass
        )

        impact = (
            closure_impact_for_edge(
                edge=edge,

                module_pairs=(
                    module_pairs
                ),

                train_pairs=(
                    train_pairs
                ),

                module_closure=(
                    module_closure
                ),

                train_closure=(
                    train_closure
                ),
            )
        )

        row = {

            "cosine":
                scores.get(
                    edge
                ),

            "subclass":
                subclass,

            "subclass_label":
                get_label(
                    subclass,
                    train_graph,
                    valid_graph,
                    module_graph,
                ),

            "superclass":
                superclass,

            "superclass_label":
                get_label(
                    superclass,
                    train_graph,
                    valid_graph,
                    module_graph,
                ),

            "sub_mowl_total_occurrences":
                sub[
                    "total_occurrences"
                ],

            "sub_mowl_unique_training_rows":
                sub[
                    "unique_training_rows"
                ],

            "sub_mowl_lhs_occurrences":
                sub[
                    "lhs_occurrences"
                ],

            "sub_mowl_rhs_occurrences":
                sub[
                    "rhs_occurrences"
                ],

            "sub_mowl_gci_types_present":
                sub[
                    "gci_types_present"
                ],

            "sub_gci0_rows":
                sub.get(
                    "gci0_rows",
                    0,
                ),

            "sub_gci1_rows":
                sub.get(
                    "gci1_rows",
                    0,
                ),

            "sub_gci2_rows":
                sub.get(
                    "gci2_rows",
                    0,
                ),

            "sub_gci3_rows":
                sub.get(
                    "gci3_rows",
                    0,
                ),

            "sub_gci0_bot_rows":
                sub.get(
                    "gci0_bot_rows",
                    0,
                ),

            "sub_gci1_bot_rows":
                sub.get(
                    "gci1_bot_rows",
                    0,
                ),

            "sub_gci3_bot_rows":
                sub.get(
                    "gci3_bot_rows",
                    0,
                ),

            "super_mowl_total_occurrences":
                sup[
                    "total_occurrences"
                ],

            "super_mowl_unique_training_rows":
                sup[
                    "unique_training_rows"
                ],

            "super_mowl_lhs_occurrences":
                sup[
                    "lhs_occurrences"
                ],

            "super_mowl_rhs_occurrences":
                sup[
                    "rhs_occurrences"
                ],

            "super_mowl_gci_types_present":
                sup[
                    "gci_types_present"
                ],

            **impact,

            "zero_subclass_mowl_signal":
                (
                    sub[
                        "unique_training_rows"
                    ]
                    ==
                    0
                ),

            "zero_superclass_mowl_signal":
                (
                    sup[
                        "unique_training_rows"
                    ]
                    ==
                    0
                ),
        }

        rows.append(
            row
        )

    # --------------------------------------------------------
    # If scores exist, sort worst -> best.
    # Otherwise sort by closure impact and then mOWL support.
    # --------------------------------------------------------

    if scores:

        rows.sort(
            key=lambda row:
                row[
                    "cosine"
                ]
        )

    else:

        rows.sort(
            key=lambda row: (
                -
                row[
                    "marginal_closure_pairs_restored"
                ],

                row[
                    "sub_mowl_unique_training_rows"
                ],
            )
        )

    # ========================================================
    # GROUPS
    # ========================================================

    zero_subclass = [
        row
        for row in rows
        if row[
            "zero_subclass_mowl_signal"
        ]
    ]

    weak_subclass = [
        row
        for row in rows
        if (
            0
            <
            row[
                "sub_mowl_unique_training_rows"
            ]
            <=
            2
        )
    ]

    scored_rows = [
        row
        for row in rows
        if row[
            "cosine"
        ]
        is not None
    ]

    poor_rows = [
        row
        for row in scored_rows
        if row[
            "cosine"
        ]
        <
        0.20
    ]

    # ========================================================
    # CORRELATIONS
    # ========================================================

    correlations = {}

    if scored_rows:

        cosines = [
            row[
                "cosine"
            ]
            for row in scored_rows
        ]

        metrics = {

            "sub_mowl_total_occurrences":
                [
                    row[
                        "sub_mowl_total_occurrences"
                    ]
                    for row in scored_rows
                ],

            "sub_mowl_unique_training_rows":
                [
                    row[
                        "sub_mowl_unique_training_rows"
                    ]
                    for row in scored_rows
                ],

            "sub_mowl_lhs_occurrences":
                [
                    row[
                        "sub_mowl_lhs_occurrences"
                    ]
                    for row in scored_rows
                ],

            "sub_mowl_rhs_occurrences":
                [
                    row[
                        "sub_mowl_rhs_occurrences"
                    ]
                    for row in scored_rows
                ],

            "individual_closure_pairs_lost":
                [
                    row[
                        "individual_closure_pairs_lost"
                    ]
                    for row in scored_rows
                ],

            "marginal_closure_pairs_restored":
                [
                    row[
                        "marginal_closure_pairs_restored"
                    ]
                    for row in scored_rows
                ],
        }

        for name, values in metrics.items():

            correlations[
                f"pearson_cos_vs_{name}"
            ] = pearson(
                values,
                cosines,
            )

            correlations[
                f"spearman_cos_vs_{name}"
            ] = spearman(
                values,
                cosines,
            )

    # ========================================================
    # TABLE
    # ========================================================

    print("=" * 78)

    print(
        "VALIDATION CLASS MOWL COVERAGE"
    )

    print("=" * 78)

    print()

    print(
        "Cosine   Rows Occ LHS RHS "
        "G0 G1 G2 G3  Lost Restore  "
        "Subclass -> Superclass"
    )

    print(
        "-" * 145
    )

    text_lines = []

    for row in rows:

        if row[
            "cosine"
        ] is None:

            cosine_text = "   N/A "

        else:

            cosine_text = (
                f"{row['cosine']:+.4f}"
            )

        line = (

            f"{cosine_text:>8} "

            f"{row['sub_mowl_unique_training_rows']:>4} "

            f"{row['sub_mowl_total_occurrences']:>3} "

            f"{row['sub_mowl_lhs_occurrences']:>3} "

            f"{row['sub_mowl_rhs_occurrences']:>3} "

            f"{row['sub_gci0_rows']:>2} "

            f"{row['sub_gci1_rows']:>2} "

            f"{row['sub_gci2_rows']:>2} "

            f"{row['sub_gci3_rows']:>2} "

            f"{row['individual_closure_pairs_lost']:>5} "

            f"{row['marginal_closure_pairs_restored']:>7}  "

            f"{row['subclass_label']} "

            f"-> "

            f"{row['superclass_label']}"
        )

        print(
            line
        )

        text_lines.append(
            line
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print("=" * 78)

    print(
        "SUMMARY"
    )

    print("=" * 78)

    print()

    print(
        f"Validation pairs                 : "
        f"{len(rows)}"
    )

    print(
        f"Zero subclass mOWL signal        : "
        f"{len(zero_subclass)}"
    )

    print(
        f"Only 1-2 normalized train rows   : "
        f"{len(weak_subclass)}"
    )

    if scores:

        print(
            f"Poor cosine pairs (< 0.20)       : "
            f"{len(poor_rows)}"
        )

    else:

        print(
            "Poor cosine pairs                : "
            "N/A (no score file)"
        )

    print()

    print(
        f"Full module closure              : "
        f"{len(module_closure)}"
    )

    print(
        f"Current train closure            : "
        f"{len(train_closure)}"
    )

    print(
        f"Total closure relationships lost : "
        f"{len(total_closure_lost)}"
    )

    print(
        f"Closure retention                : "
        f"{retention:.1%}"
    )

    print()

    # ========================================================
    # ZERO MOWL SIGNAL
    # ========================================================

    if zero_subclass:

        print(
            "Validation subclasses with ZERO actual "
            "mOWL training rows:"
        )

        print()

        for row in zero_subclass:

            print(
                f"  "
                f"{row['subclass_label']} "
                f"-> "
                f"{row['superclass_label']}"
            )

        print()

    # ========================================================
    # HIGH-DAMAGE VALIDATION EDGES
    # ========================================================

    print(
        "Validation edges with greatest marginal "
        "closure restoration:"
    )

    print()

    for row in sorted(
        rows,
        key=lambda item:
            item[
                "marginal_closure_pairs_restored"
            ],
        reverse=True,
    )[:10]:

        print(
            f"  restore="
            f"{row['marginal_closure_pairs_restored']:<3} "
            f"lost="
            f"{row['individual_closure_pairs_lost']:<3} "
            f"rows="
            f"{row['sub_mowl_unique_training_rows']:<3} "
            f"{row['subclass_label']} "
            f"-> "
            f"{row['superclass_label']}"
        )

    print()

    # ========================================================
    # POOR SCORE DETAILS
    # ========================================================

    if poor_rows:

        print(
            "Poor validation pairs and actual "
            "mOWL support:"
        )

        print()

        for row in poor_rows:

            print(
                f"  {row['cosine']:+.4f}  "
                f"rows="
                f"{row['sub_mowl_unique_training_rows']:<3} "
                f"occ="
                f"{row['sub_mowl_total_occurrences']:<3} "
                f"lhs="
                f"{row['sub_mowl_lhs_occurrences']:<3} "
                f"rhs="
                f"{row['sub_mowl_rhs_occurrences']:<3} "
                f"closure_lost="
                f"{row['individual_closure_pairs_lost']:<3} "
                f"restore="
                f"{row['marginal_closure_pairs_restored']:<3} "
                f"{row['subclass_label']} "
                f"-> "
                f"{row['superclass_label']}"
            )

        print()

    # ========================================================
    # CORRELATIONS
    # ========================================================

    if correlations:

        print(
            "Correlation with validation cosine:"
        )

        print()

        for name, value in correlations.items():

            display = (
                "N/A"
                if value is None
                else f"{value:+.4f}"
            )

            print(
                f"  {name:<58}"
                f"{display}"
            )

        print()

    else:

        print(
            "Correlation analysis             : "
            "SKIPPED (no validation scores)"
        )

        print()

    # ========================================================
    # SAVE CSV
    # ========================================================

    save_csv(
        rows
    )

    # ========================================================
    # JSON
    # ========================================================

    report = {

        "script_version":
            SCRIPT_VERSION,

        "mode":
            (
                "scored"
                if scores
                else "pre_training"
            ),

        "score_file":
            (
                str(
                    args.scores_file
                )
                if args.scores_file
                is not None
                else None
            ),

        "ontology": {

            "module_subclass_pairs":
                len(
                    module_pairs
                ),

            "train_subclass_pairs":
                len(
                    train_pairs
                ),

            "validation_subclass_pairs":
                len(
                    valid_pairs
                ),

            "module_closure":
                len(
                    module_closure
                ),

            "training_closure":
                len(
                    train_closure
                ),

            "total_closure_pairs_lost":
                len(
                    total_closure_lost
                ),

            "closure_retention":
                retention,

            "indexed_classes":
                len(
                    dataset.classes.as_str
                ),

            "root_axioms":
                int(
                    root_axioms.size()
                ),

            "imported_logical_axioms":
                False,

            "imported_vocabulary_retained":
                True,
        },

        "training_datasets":
            dataset_summary,

        "summary": {

            "validation_pairs":
                len(
                    rows
                ),

            "zero_subclass_mowl_signal":
                len(
                    zero_subclass
                ),

            "weak_subclass_mowl_signal":
                len(
                    weak_subclass
                ),

            "poor_cosine_pairs":
                (
                    len(
                        poor_rows
                    )
                    if scores
                    else None
                ),
        },

        "correlations":
            correlations,

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

    # ========================================================
    # TEXT REPORT
    # ========================================================

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "PROJECT 5 - ACTUAL MOWL TRAINING COVERAGE AUDIT\n"
        )

        handle.write(
            "=" * 78
            +
            "\n\n"
        )

        handle.write(
            f"Mode: "
            f"{'scored' if scores else 'pre-training'}\n"
        )

        handle.write(
            f"Module closure: "
            f"{len(module_closure)}\n"
        )

        handle.write(
            f"Training closure: "
            f"{len(train_closure)}\n"
        )

        handle.write(
            f"Closure lost: "
            f"{len(total_closure_lost)}\n"
        )

        handle.write(
            f"Closure retention: "
            f"{retention:.1%}\n"
        )

        handle.write(
            f"Zero mOWL subclass support: "
            f"{len(zero_subclass)}\n\n"
        )

        handle.write(
            "Validation pairs\n"
        )

        handle.write(
            "-" * 78
            +
            "\n"
        )

        for line in text_lines:

            handle.write(
                line
                +
                "\n"
            )

        if correlations:

            handle.write(
                "\nCorrelations\n"
            )

            handle.write(
                "-" * 78
                +
                "\n"
            )

            for name, value in correlations.items():

                handle.write(
                    f"{name}: {value}\n"
                )

    print(
        "Reports written:"
    )

    print(
        f"  CSV  : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"  JSON : "
        f"{OUTPUT_JSON}"
    )

    print(
        f"  TXT  : "
        f"{OUTPUT_TXT}"
    )

    print()

    print(
        "No model training was performed."
    )


if __name__ == "__main__":

    main()