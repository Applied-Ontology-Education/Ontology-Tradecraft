import json
import random
from pathlib import Path

from rdflib import Graph, OWL, RDF, RDFS, URIRef

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODULE_PATH = PROJECT_DIR / "src" / "cco-module.ttl"
TRAIN_PATH = PROJECT_DIR / "src" / "train.ttl"
VALID_PATH = PROJECT_DIR / "src" / "valid.ttl"
SPLIT_REPORT_PATH = PROJECT_DIR / "reports" / "split_summary.json"

VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

DECLARATION_TYPES = {
    OWL.Class,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
}


def is_named_subclass(triple):
    subject, predicate, obj = triple
    return (
        predicate == RDFS.subClassOf
        and isinstance(subject, URIRef)
        and isinstance(obj, URIRef)
    )


def transitive_closure(asserted_pairs):
    parents = {}
    for subclass, superclass in asserted_pairs:
        parents.setdefault(subclass, set()).add(superclass)

    def ancestors(cls):
        seen = set()
        stack = list(parents.get(cls, ()))
        while stack:
            current = stack.pop()
            if current in seen or current == cls:
                continue
            seen.add(current)
            stack.extend(parents.get(current, ()))
        return seen

    return sorted(
        {(cls, ancestor) for cls in parents for ancestor in ancestors(cls)}
    )


def split_closure(closure_pairs):
    rng = random.Random(RANDOM_SEED)
    target = round(VALIDATION_FRACTION * len(closure_pairs))

    remaining_per_subclass = {}
    for subclass, _ in closure_pairs:
        remaining_per_subclass[subclass] = remaining_per_subclass.get(subclass, 0) + 1

    shuffled = closure_pairs[:]
    rng.shuffle(shuffled)

    validation = []
    for subclass, superclass in shuffled:
        if len(validation) >= target:
            break
        # Never remove a subclass's last training pair.
        if remaining_per_subclass[subclass] > 1:
            validation.append((subclass, superclass))
            remaining_per_subclass[subclass] -= 1

    validation_set = set(validation)
    training = [pair for pair in closure_pairs if pair not in validation_set]
    return training, sorted(validation)


def main():
    module = Graph()
    module.parse(str(MODULE_PATH), format="turtle")

    asserted = sorted(
        (s, o) for s, _, o in module if is_named_subclass((s, _, o))
    )
    closure = transitive_closure(asserted)
    training, validation = split_closure(closure)

    # train.ttl: everything in the module except named subClassOf triples,
    # plus the training closure pairs.
    train_graph = Graph()
    for prefix, namespace in module.namespaces():
        train_graph.bind(prefix, namespace)
    for triple in module:
        if not is_named_subclass(triple):
            train_graph.add(triple)
    for subclass, superclass in training:
        train_graph.add((subclass, RDFS.subClassOf, superclass))
    train_graph.serialize(str(TRAIN_PATH), format="turtle")

    # valid.ttl: all declarations + held-out pairs.
    valid_graph = Graph()
    for prefix, namespace in module.namespaces():
        valid_graph.bind(prefix, namespace)
    for subject, _, obj in module.triples((None, RDF.type, None)):
        if obj in DECLARATION_TYPES:
            valid_graph.add((subject, RDF.type, obj))
    for subclass, superclass in validation:
        valid_graph.add((subclass, RDF.type, OWL.Class))
        valid_graph.add((superclass, RDF.type, OWL.Class))
        valid_graph.add((subclass, RDFS.subClassOf, superclass))
    valid_graph.serialize(str(VALID_PATH), format="turtle")

    asserted_set = set(asserted)
    summary = {
        "method": "deductive-closure split; every held-out subclass keeps >=1 training pair",
        "asserted_subclass_axioms": len(asserted),
        "closure_subclass_pairs": len(closure),
        "training_pairs": len(training),
        "validation_pairs": len(validation),
        "validation_fraction": round(len(validation) / len(closure), 3),
        "validation_pairs_that_are_asserted": sum(p in asserted_set for p in validation),
        "random_seed": RANDOM_SEED,
    }
    SPLIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nONTOLOGY SPLIT")
    print("---------------------------------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()