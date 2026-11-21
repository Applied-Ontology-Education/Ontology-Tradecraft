from pathlib import Path
from collections import defaultdict, deque
import json

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL

SCRIPT_VERSION = "project-5-frozen-clean-room-split-v6"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODULE_FILE = SRC_DIR / "module.ttl"
TRAIN_FILE = SRC_DIR / "train.ttl"
VALID_FILE = SRC_DIR / "valid.ttl"
REPORT_FILE = REPORTS_DIR / "split_train_valid_report.json"

EXPECTED_MODULE_PAIRS = 103
EXPECTED_TRAIN_PAIRS = 82
EXPECTED_VALID_PAIRS = 21
EXPECTED_FULL_CLOSURE = 225
EXPECTED_TRAIN_CLOSURE = 169
MIN_CLOSURE_RETENTION = 0.75

CCO = "https://www.commoncoreontologies.org/"
BFO = "http://purl.obolibrary.org/obo/"

# Frozen validation split used by the successful v8 mOWL run.
# Labels are resolved from module.ttl at runtime; BFO/CCO identifiers
# are given explicitly where needed.
FROZEN_VALIDATION_SPEC = [
    ("Military Personnel Force", "Armed Force"),
    ("Division of Delimiting Domain", CCO + "ont00000472"),
    ("Agent Capability", BFO + "BFO_0000017"),
    ("Process Regulation", CCO + "ont00000965"),
    ("Ethnic Group", "Group of Persons"),
    ("Objective", CCO + "ont00000965"),
    ("Material Territory of a Government Domain", CCO + "ont00001341"),
    ("Skill", "Agent Capability"),
    ("Fourth-Order Administrative Region", "Government Domain"),
    ("Agent", BFO + "BFO_0000040"),
    ("Eye", "Bodily Component"),
    ("Commercial Role", BFO + "BFO_0000023"),
    ("Neutral Role", "Allegiance Role"),
    ("Geopolitical Power Role", BFO + "BFO_0000023"),
    ("Commercial Organization", "Organization"),
    ("Government Organization", "Organization"),
    ("Domain of a Country", "Government Domain"),
    ("Process Prohibition", "Process Regulation"),
    ("Organization Member", "Person"),
    ("Government", "Organization"),
    ("Government of a Country", "Government"),
]


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n  {path}")


def load_graph(path: Path) -> Graph:
    require_file(path)
    graph = Graph()
    graph.parse(str(path), format="turtle")
    return graph


def named_subclass_pairs(graph: Graph):
    pairs = set()

    for child, parent in graph.subject_objects(RDFS.subClassOf):
        if (
            isinstance(child, URIRef)
            and isinstance(parent, URIRef)
            and child != parent
        ):
            pairs.add(
                (
                    str(child),
                    str(parent),
                )
            )

    return pairs


def build_label_index(graph: Graph):
    index = defaultdict(list)

    for subject, label in graph.subject_objects(RDFS.label):
        if isinstance(subject, URIRef):
            text = str(label).strip()

            if text:
                index[text].append(
                    str(subject)
                )

    return index


def resolve_class_reference(
    value: str,
    label_index,
):
    if (
        value.startswith("http://")
        or value.startswith("https://")
    ):
        return value

    matches = label_index.get(
        value,
        [],
    )

    if not matches:
        raise RuntimeError(
            "Could not resolve class label from module.ttl:\n"
            f"  {value}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Class label is ambiguous in module.ttl:\n"
            f"  {value}\n"
            "Matching IRIs:\n"
            + "\n".join(
                f"  {iri}"
                for iri in matches
            )
        )

    return matches[0]


def transitive_closure_pairs(pairs):
    adjacency = defaultdict(set)
    nodes = set()

    for child, parent in pairs:
        adjacency[child].add(parent)
        nodes.add(child)
        nodes.add(parent)

    closure = set()

    for start in sorted(nodes):
        queue = deque(
            sorted(
                adjacency.get(
                    start,
                    set(),
                )
            )
        )

        visited = set()

        while queue:
            current = queue.popleft()

            if current in visited:
                continue

            visited.add(current)

            if current != start:
                closure.add(
                    (
                        start,
                        current,
                    )
                )

            for parent in sorted(
                adjacency.get(
                    current,
                    set(),
                )
            ):
                if parent not in visited:
                    queue.append(parent)

    return closure


def pair_is_entailed(
    pair,
    training_pairs,
):
    child, target_parent = pair

    adjacency = defaultdict(set)

    for left, right in training_pairs:
        adjacency[left].add(right)

    queue = deque(
        sorted(
            adjacency.get(
                child,
                set(),
            )
        )
    )

    visited = set()

    while queue:
        current = queue.popleft()

        if current == target_parent:
            return True

        if current in visited:
            continue

        visited.add(current)

        for parent in sorted(
            adjacency.get(
                current,
                set(),
            )
        ):
            if parent not in visited:
                queue.append(parent)

    return False


def copy_namespaces(
    source: Graph,
    destination: Graph,
):
    for prefix, namespace in source.namespaces():
        destination.bind(
            prefix,
            namespace,
        )


def build_valid_graph(
    module_graph: Graph,
    validation_pairs,
):
    valid_graph = Graph()

    copy_namespaces(
        module_graph,
        valid_graph,
    )

    declaration_types = {
        OWL.Class,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
        OWL.NamedIndividual,
        RDFS.Datatype,
    }

    ontology_nodes = set(
        module_graph.subjects(
            RDF.type,
            OWL.Ontology,
        )
    )

    for (
        subject,
        predicate,
        obj,
    ) in module_graph:

        if (
            predicate == RDF.type
            and obj in declaration_types
        ):
            valid_graph.add(
                (
                    subject,
                    predicate,
                    obj,
                )
            )

        elif (
            predicate == RDF.type
            and obj == OWL.Ontology
        ):
            valid_graph.add(
                (
                    subject,
                    predicate,
                    obj,
                )
            )

        elif (
            predicate == OWL.imports
            and subject in ontology_nodes
        ):
            valid_graph.add(
                (
                    subject,
                    predicate,
                    obj,
                )
            )

    for child, parent in sorted(
        validation_pairs
    ):
        valid_graph.add(
            (
                URIRef(child),
                RDFS.subClassOf,
                URIRef(parent),
            )
        )

    return valid_graph


def main():
    print(
        f"SPLIT_VERSION           : "
        f"{SCRIPT_VERSION}"
    )

    print()

    print("=" * 78)
    print(
        "PROJECT 5 - CLEAN-ROOM "
        "TRAIN / VALIDATION SPLIT"
    )
    print("=" * 78)

    print()

    print(
        f"Module                  : "
        f"{MODULE_FILE}"
    )

    print(
        f"Training output         : "
        f"{TRAIN_FILE}"
    )

    print(
        f"Validation output       : "
        f"{VALID_FILE}"
    )

    print()

    # ========================================================
    # LOAD MODULE.TTL
    # ========================================================

    module_graph = load_graph(
        MODULE_FILE
    )

    module_pairs = named_subclass_pairs(
        module_graph
    )

    if (
        len(module_pairs)
        != EXPECTED_MODULE_PAIRS
    ):
        raise RuntimeError(
            "Unexpected named subclass count "
            "in module.ttl.\n"
            f"Expected: "
            f"{EXPECTED_MODULE_PAIRS}\n"
            f"Actual:   "
            f"{len(module_pairs)}"
        )

    # ========================================================
    # RESOLVE VALIDATION AXIOMS FROM MODULE.TTL
    # ========================================================

    label_index = build_label_index(
        module_graph
    )

    validation_pairs = set()

    for (
        child_ref,
        parent_ref,
    ) in FROZEN_VALIDATION_SPEC:

        child = resolve_class_reference(
            child_ref,
            label_index,
        )

        parent = resolve_class_reference(
            parent_ref,
            label_index,
        )

        pair = (
            child,
            parent,
        )

        if pair not in module_pairs:
            raise RuntimeError(
                "Frozen validation axiom "
                "was not found in module.ttl:\n"
                f"  {child_ref} -> {parent_ref}\n"
                "Resolved as:\n"
                f"  {child} -> {parent}"
            )

        validation_pairs.add(
            pair
        )

    if (
        len(validation_pairs)
        != EXPECTED_VALID_PAIRS
    ):
        raise RuntimeError(
            "Validation specification did not "
            "resolve to exactly "
            f"{EXPECTED_VALID_PAIRS} unique axioms.\n"
            f"Actual: "
            f"{len(validation_pairs)}"
        )

    # ========================================================
    # TRAINING = MODULE MINUS VALIDATION
    # ========================================================

    training_pairs = (
        module_pairs
        -
        validation_pairs
    )

    if (
        len(training_pairs)
        != EXPECTED_TRAIN_PAIRS
    ):
        raise RuntimeError(
            "Training split size is incorrect.\n"
            f"Expected: "
            f"{EXPECTED_TRAIN_PAIRS}\n"
            f"Actual:   "
            f"{len(training_pairs)}"
        )

    # ========================================================
    # CHECK OVERLAP / COMPLETENESS
    # ========================================================

    overlap = (
        training_pairs
        &
        validation_pairs
    )

    if overlap:
        raise RuntimeError(
            "Direct train/validation overlap "
            f"detected: {len(overlap)}"
        )

    if (
        training_pairs
        |
        validation_pairs
        !=
        module_pairs
    ):
        raise RuntimeError(
            "Training plus validation does not "
            "reconstruct the module subclass axioms."
        )

    # ========================================================
    # CHECK TRANSITIVE LEAKAGE
    # ========================================================

    leaked = [
        pair

        for pair
        in sorted(
            validation_pairs
        )

        if pair_is_entailed(
            pair,
            training_pairs,
        )
    ]

    if leaked:
        raise RuntimeError(
            "Validation leakage detected:\n"
            +
            "\n".join(
                f"  {child} -> {parent}"
                for child, parent
                in leaked
            )
        )

    # ========================================================
    # CLOSURE CHECK
    # ========================================================

    full_closure = (
        transitive_closure_pairs(
            module_pairs
        )
    )

    train_closure = (
        transitive_closure_pairs(
            training_pairs
        )
    )

    retention = (
        len(train_closure)
        /
        len(full_closure)
        if full_closure
        else 1.0
    )

    if (
        len(full_closure)
        != EXPECTED_FULL_CLOSURE
    ):
        raise RuntimeError(
            "Unexpected full-module "
            "closure size.\n"
            f"Expected: "
            f"{EXPECTED_FULL_CLOSURE}\n"
            f"Actual:   "
            f"{len(full_closure)}"
        )

    if (
        len(train_closure)
        != EXPECTED_TRAIN_CLOSURE
    ):
        raise RuntimeError(
            "The deterministic split did not "
            "reproduce the validated training closure.\n"
            f"Expected: "
            f"{EXPECTED_TRAIN_CLOSURE}\n"
            f"Actual:   "
            f"{len(train_closure)}\n"
            f"Retention: "
            f"{retention * 100:.1f}%"
        )

    if (
        retention
        <
        MIN_CLOSURE_RETENTION
    ):
        raise RuntimeError(
            "Closure retention is below "
            "the required floor.\n"
            f"Retention: "
            f"{retention * 100:.1f}%"
        )

    # ========================================================
    # WRITE TRAIN.TTL
    # ========================================================

    validation_triples = {
        (
            URIRef(child),
            RDFS.subClassOf,
            URIRef(parent),
        )

        for child, parent
        in validation_pairs
    }

    train_graph = Graph()

    copy_namespaces(
        module_graph,
        train_graph,
    )

    for triple in module_graph:
        if triple not in validation_triples:
            train_graph.add(
                triple
            )

    # ========================================================
    # WRITE VALID.TTL
    # ========================================================

    valid_graph = build_valid_graph(
        module_graph,
        validation_pairs,
    )

    SRC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_graph.serialize(
        destination=str(
            TRAIN_FILE
        ),
        format="turtle",
    )

    valid_graph.serialize(
        destination=str(
            VALID_FILE
        ),
        format="turtle",
    )

    # ========================================================
    # VERIFY WRITTEN FILES
    # ========================================================

    written_train = named_subclass_pairs(
        load_graph(
            TRAIN_FILE
        )
    )

    written_valid = named_subclass_pairs(
        load_graph(
            VALID_FILE
        )
    )

    if (
        written_train
        !=
        training_pairs
    ):
        raise RuntimeError(
            "Written train.ttl does not match "
            "the calculated training split."
        )

    if (
        written_valid
        !=
        validation_pairs
    ):
        raise RuntimeError(
            "Written valid.ttl does not match "
            "the calculated validation split."
        )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "script_version":
            SCRIPT_VERSION,

        "status":
            "PASS",

        "source_module":
            str(
                MODULE_FILE
            ),

        "module_named_subclass_axioms":
            len(
                module_pairs
            ),

        "training_named_subclass_axioms":
            len(
                training_pairs
            ),

        "validation_named_subclass_axioms":
            len(
                validation_pairs
            ),

        "direct_overlap":
            0,

        "entailed_validation_axioms":
            0,

        "full_module_closure":
            len(
                full_closure
            ),

        "training_closure":
            len(
                train_closure
            ),

        "closure_retention":
            retention,

        "train_file":
            str(
                TRAIN_FILE
            ),

        "valid_file":
            str(
                VALID_FILE
            ),

        "validation_pairs":
            [
                {
                    "subclass":
                        child,

                    "superclass":
                        parent,
                }

                for child, parent
                in sorted(
                    validation_pairs
                )
            ],
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2,
        )

    print(
        f"Named subclass axioms   : "
        f"{len(module_pairs)}"
    )

    print(
        f"Training axioms         : "
        f"{len(training_pairs)}"
    )

    print(
        f"Validation axioms       : "
        f"{len(validation_pairs)}"
    )

    print(
        f"Full module closure     : "
        f"{len(full_closure)}"
    )

    print(
        f"Training closure        : "
        f"{len(train_closure)}"
    )

    print(
        f"Closure retention       : "
        f"{retention * 100:.1f}%"
    )

    print(
        "Direct overlap          : 0"
    )

    print(
        "Validation leakage      : 0"
    )

    print(
        "Split validation        : PASS"
    )

    print()

    print(
        f"Training ontology       : "
        f"{TRAIN_FILE}"
    )

    print(
        f"Validation ontology     : "
        f"{VALID_FILE}"
    )

    print(
        f"Split report            : "
        f"{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()