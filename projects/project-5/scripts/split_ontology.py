import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import rdflib
import yaml
from rdflib import OWL, RDF, RDFS, BNode, URIRef

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"
CCO = "https://www.commoncoreontologies.org/"


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_graph(path):
    g = rdflib.Graph()
    g.parse(path, format="turtle")
    return g


def collect_fillers(graph, node, out):
    for p, o in graph.predicate_objects(node):
        if isinstance(o, BNode):
            collect_fillers(graph, o, out)
        elif p in (OWL.someValuesFrom, RDF.first):
            out.add(o)


def classes_in_other_axioms(module):
    found = set()
    for a, b in module.subject_objects(RDFS.subClassOf):
        if isinstance(b, BNode):
            found.add(a)
            collect_fillers(module, b, found)
    for a, b in module.subject_objects(OWL.equivalentClass):
        found.add(a)
        collect_fillers(module, b, found)
    for a, b in module.subject_objects(OWL.disjointWith):
        found.update((a, b))
    return found


def still_anchored(cls, held_out, named_pairs, other_axiom_classes):
    if cls in other_axiom_classes:
        return True
    return any(cls in pair for pair in named_pairs if pair not in held_out)


def choose_held_out(pool, named_pairs, other_axiom_classes, n_target, seed):
    order = sorted(pool)
    random.Random(seed).shuffle(order)
    held_out = set()
    for pair in order:
        if len(held_out) == n_target:
            break
        trial = held_out | {pair}
        classes = {c for p in trial for c in p}
        if all(still_anchored(c, trial, named_pairs, other_axiom_classes) for c in classes):
            held_out = trial
    return held_out


def undeclared_properties(module, bfo):
    used = set(module.objects(None, OWL.onProperty))
    declared = set(module.subjects(RDF.type, OWL.ObjectProperty))
    declared |= set(bfo.subjects(RDF.type, OWL.ObjectProperty))
    return used - declared


def build_graphs(module, bfo, held_out, missing_props):
    train = rdflib.Graph()
    for prefix, ns in module.namespaces():
        train.bind(prefix, ns)
    train += module
    for a, b in held_out:
        train.remove((a, RDFS.subClassOf, b))
    train.remove((None, OWL.imports, None))

    bfo_header = next(bfo.subjects(RDF.type, OWL.Ontology))
    for triple in bfo:
        if triple[0] != bfo_header:
            train.add(triple)

    for p in missing_props:
        train.add((p, RDF.type, OWL.ObjectProperty))

    valid = rdflib.Graph()
    for prefix, ns in module.namespaces():
        valid.bind(prefix, ns)
    module_classes = {c for c in module.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)}
    held_classes = {c for pair in held_out for c in pair}
    for c in module_classes | held_classes:
        valid.add((c, RDF.type, OWL.Class))
        for label in module.objects(c, RDFS.label):
            valid.add((c, RDFS.label, label))
        for label in bfo.objects(c, RDFS.label):
            valid.add((c, RDFS.label, label))
    for a, b in held_out:
        valid.add((a, RDFS.subClassOf, b))

    return train, valid


def elk_entailed(train_path, held_out):
    robot = shutil.which("robot")
    if robot is None:
        sys.exit("robot not found on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        inferred_path = Path(tmp) / "train_inferred.ttl"
        subprocess.run(
            [robot, "reason", "--reasoner", "ELK",
             "--input", str(train_path),
             "--axiom-generators", "SubClass",
             "--include-indirect", "true",
             # ROBOT's default (true) deletes the indirect links again
             "--remove-redundant-subclass-axioms", "false",
             "--output", str(inferred_path)],
            check=True,
        )
        inferred = load_graph(inferred_path)
    return {(a, b) for a, b in held_out if (a, RDFS.subClassOf, b) in inferred}


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    cfg = load_config()
    module_path = PROJECT_DIR / cfg["paths"]["module"]
    bfo_path = PROJECT_DIR / cfg["paths"]["bfo"]

    module = load_graph(module_path)
    bfo = load_graph(bfo_path)

    print(f"module: {len(module)} triples from {module_path.name}")
    print(f"bfo:    {len(bfo)} triples from {bfo_path.name}")

    named_pairs = [
        (a, b)
        for a, b in module.subject_objects(RDFS.subClassOf)
        if isinstance(a, URIRef) and isinstance(b, URIRef)
    ]

    equiv_defined = set(module.subjects(OWL.equivalentClass, None))

    pool = [
        (a, b)
        for a, b in named_pairs
        if str(a).startswith(CCO) and a not in equiv_defined
    ]

    print(f"named subclass pairs: {len(named_pairs)}")
    print(f"equivalence-defined:  {len(equiv_defined)}")
    print(f"pool:                 {len(pool)}")

    other_axiom_classes = classes_in_other_axioms(module)
    print(f"classes in other axioms: {len(other_axiom_classes)}")

    n_target = round(cfg["split"]["valid_fraction"] * len(named_pairs))
    held_out = choose_held_out(
        pool, named_pairs, other_axiom_classes, n_target, cfg["split"]["seed"]
    )
    if len(held_out) < n_target:
        sys.exit(f"only {len(held_out)} of {n_target} pairs could be held out")

    print(f"held out: {len(held_out)} of target {n_target}")
    print(sorted(held_out)[:3])

    missing_props = undeclared_properties(module, bfo)
    print(f"undeclared properties: {sorted(str(p) for p in missing_props)}")

    train, valid = build_graphs(module, bfo, held_out, missing_props)
    print(f"train: {len(train)} triples")
    print(f"valid: {len(valid)} triples")

    train_path = PROJECT_DIR / cfg["paths"]["train"]
    valid_path = PROJECT_DIR / cfg["paths"]["valid"]
    train.serialize(train_path, format="turtle")
    valid.serialize(valid_path, format="turtle")
    print(f"wrote {train_path.name} and {valid_path.name}")

    moved_back = elk_entailed(train_path, held_out)
    if moved_back:
        held_out -= moved_back
        train, valid = build_graphs(module, bfo, held_out, missing_props)
        train.serialize(train_path, format="turtle")
        valid.serialize(valid_path, format="turtle")
    print(f"moved back by ELK: {len(moved_back)}")

    bfo_header = next(bfo.subjects(RDF.type, OWL.Ontology))
    report = {
        "seed": cfg["split"]["seed"],
        "valid_fraction": cfg["split"]["valid_fraction"],
        "named_subclass_pairs": len(named_pairs),
        "pool": len(pool),
        "target": n_target,
        "moved_back_by_elk": len(moved_back),
        "held_out": len(held_out),
        "pool_rule": "named A subClassOf B pairs with a CCO-namespace subclass "
                     "that has no owl:equivalentClass definition",
        "eligibility_rule": "every class in every held-out pair still appears in a "
                            "training axiom: another named subclass pair, an existential "
                            "restriction (subject or filler), an equivalence, or a disjointness",
        "bfo_version_iri": str(bfo.value(bfo_header, OWL.versionIRI)),
        "declared_properties": sorted(str(p) for p in missing_props),
        "moved_back_pairs": sorted([str(a), str(b)] for a, b in moved_back),
    }
    report_path = PROJECT_DIR / cfg["paths"]["split_report"]
    write_report(report_path, report)
    print(f"wrote {report_path.name}")


if __name__ == "__main__":
    main()
