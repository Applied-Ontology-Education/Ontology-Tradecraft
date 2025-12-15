#!/usr/bin/env python3
"""
Create train.ttl and valid.ttl splits from a source ontology.

Goal (per MOWL PathDataset and ELEmbeddings examples):
- Use train ontology for learning mainly from simple named-class subclass axioms.
- Use valid ontology to evaluate on held-out named-class rdfs:subClassOf axioms.
- Both splits must preserve entity vocabulary: include all owl:Class and owl:ObjectProperty declarations
  (and optionally DataProperty declarations) in BOTH outputs so that entities exist in both graphs.
- Only partition simple named-class subclass axioms; exclude complex expressions and blank nodes.

This script implements a deterministic split with configurable ratio and seed.

Usage:
  python3 scripts/split_train_valid.py \
      --input src/InformationEntityOntology.ttl \
      --train src/train.ttl \
      --valid src/valid.ttl \
      --ratio 0.8 \
      --seed 42

Notes about MOWL docs:
- In MOWL, PathDataset(train_path, valid_path) expects two ontologies. Typical practice is to keep
  all class declarations in both and to put held-out rdfs:subClassOf axioms in the validation file.
- This script follows that pattern so ELEmbeddings can embed the same entity set across splits and
  evaluate cosine similarity for validation subClassOf pairs.
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple, Union, Dict, Set

from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, BNode, URIRef as _URIRef
from rdflib.term import Node

from common.settings import build_settings
from util.logger_config import config

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
ONTOLOGIES_ROOT = PROJECT_ROOT / "src"
IEO_FILE = ONTOLOGIES_ROOT / "InformationEntityOntology.ttl"

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)


def parse_args(settings) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split an ontology into train.ttl and valid.ttl for MOWL training")
    p.add_argument("--input", default=str(settings["reference_ontology"]), help="Source ontology TTL/OWL path")
    p.add_argument("--train", default="src/train.ttl", help="Output training TTL path")
    p.add_argument("--valid", default="src/valid.ttl", help="Output validation TTL path")
    p.add_argument("--ratio", type=float, default=0.8, help="Proportion of simple subclass axioms to put in train")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling before split")
    return p.parse_args()


def is_iri(node) -> bool:
    # rdflib URIRef behaves like a string with .startswith('http') in many ontologies
    try:
        s = str(node)
        return s.startswith("http://") or s.startswith("https://")
    except Exception:
        return False


def _is_rdf_list(g: Graph, node: Node) -> bool:
    """Return True if node appears to be an RDF list node (heuristic)."""
    if isinstance(node, BNode):
        # Typically has rdf:first or rdf:rest
        return (
            (node, RDF.first, None) in g or (node, RDF.rest, None) in g
        )
    return False


def _is_el_class_expr(g: Graph, node: Node, _seen: Set[Node] | None = None) -> bool:
    """Return True if node is an OWL EL class expression.

    Allowed:
    - Named class IRI
    - owl:Restriction with owl:onProperty IRI and owl:someValuesFrom F where F is EL expr
    - owl:intersectionOf RDF list of EL expr members
    """
    if _seen is None:
        _seen = set()
    if node in _seen:
        return True  # avoid cycles; treat as acceptable to prevent infinite loop
    _seen.add(node)

    # Named class
    if isinstance(node, _URIRef):
        return True

    # Blank node expressions
    if isinstance(node, BNode):
        # Restriction
        if (node, RDF.type, OWL.Restriction) in g:
            # onProperty must be IRI
            props = list(g.objects(node, OWL.onProperty))
            if len(props) != 1 or not all(isinstance(p, _URIRef) for p in props):
                return False
            # someValuesFrom must exist and be EL expr
            fillers = list(g.objects(node, OWL.someValuesFrom))
            if len(fillers) != 1:
                return False
            return _is_el_class_expr(g, fillers[0], _seen)

        # intersectionOf
        lists = list(g.objects(node, OWL.intersectionOf))
        if len(lists) == 1:
            head = lists[0]
            # Traverse RDF list
            cur = head
            while isinstance(cur, BNode):
                firsts = list(g.objects(cur, RDF.first))
                if len(firsts) != 1:
                    return False
                if not _is_el_class_expr(g, firsts[0], _seen):
                    return False
                rests = list(g.objects(cur, RDF.rest))
                if len(rests) != 1:
                    return False
                cur = rests[0]
            # Expect rdf:nil at end
            return isinstance(cur, _URIRef) and cur == RDF.nil

    return False


def _copy_bnode_closure(src: Graph, dst: Graph, node: BNode, mapping: Dict[BNode, BNode]) -> BNode:
    """Deep-copy the blank-node subgraph reachable from node into dst.

    Returns the corresponding new BNode in dst. Reuses nodes via mapping to
    preserve structure sharing.
    """
    if node in mapping:
        return mapping[node]
    new_node = BNode()
    mapping[node] = new_node
    for p, o in src.predicate_objects(node):
        if isinstance(o, BNode):
            o2 = _copy_bnode_closure(src, dst, o, mapping)
            dst.add((new_node, p, o2))
        else:
            dst.add((new_node, p, o))
    return new_node


def collect_el_subclass_axioms(g: Graph) -> List[Tuple[URIRef, Union[URIRef, BNode]]]:
    """Collect rdfs:subClassOf axioms where the subject is a named class and the
    object is either a named class or an OWL EL class expression (e.g., existential
    restriction, intersections of EL expressions).
    """
    axioms: List[Tuple[URIRef, Union[URIRef, BNode]]] = []
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if not is_iri(s):
            continue
        s_iri = URIRef(str(s))
        if is_iri(o):
            axioms.append((s_iri, URIRef(str(o))))
        elif isinstance(o, BNode) and _is_el_class_expr(g, o):
            axioms.append((s_iri, o))
    return axioms


def copy_prefixes(src: Graph, dst: Graph) -> None:
    # Copy all namespace bindings for nicer TTL output
    for prefix, ns in src.namespace_manager.namespaces():
        try:
            dst.namespace_manager.bind(prefix, Namespace(str(ns)), override=False)
        except Exception:
            # Ignore binding errors, rdflib will still serialize with URIs
            pass


def add_declarations_to(graph: Graph, source: Graph) -> int:
    count = 0
    # Classes
    for c in set(source.subjects(RDF.type, OWL.Class)):
        graph.add((c, RDF.type, OWL.Class))
        count += 1
    # Object properties
    for p in set(source.subjects(RDF.type, OWL.ObjectProperty)):
        graph.add((p, RDF.type, OWL.ObjectProperty))
        count += 1
    # Data properties (optional, but keep for completeness)
    for p in set(source.subjects(RDF.type, OWL.DatatypeProperty)):
        graph.add((p, RDF.type, OWL.DatatypeProperty))
        count += 1
    return count


def main() -> None:
    # Load settings
    settings = build_settings(PROJECT_ROOT, DATA_ROOT)

    args = parse_args(settings)
    input_path = Path(args.input)
    input_path = IEO_FILE
    train_path = Path(args.train)
    valid_path = Path(args.valid)

    logger.info("Reading input ontology: %s", input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input ontology not found: {input_path}")
    input_ontology = Graph()
    input_ontology.parse(input_path.as_posix(), format="turtle")

    el_candidates_path = settings["candidates_el"]
    logger.info("Reading candidates ontology: %s", el_candidates_path)
    if not el_candidates_path.exists():
        raise FileNotFoundError(f"Candidates ontology not found: {el_candidates_path}")
    candidates_el = Graph()
    candidates_el.parse(settings["candidates_el"], format="turtle")

    # Collect EL-valid subclass axioms
    pairs = collect_el_subclass_axioms(input_ontology)
    logger.info("Found %d EL-valid subclass axioms (named or restrictions)", len(pairs))

    # Deterministic shuffle and split
    random.seed(args.seed)
    random.shuffle(pairs)
    n_train = int(len(pairs) * args.ratio)
    train_pairs = pairs[:n_train]
    valid_pairs = pairs[n_train:]

    # Build output graphs with copied prefixes and declarations
    g_train = Graph()
    g_valid = Graph()
    copy_prefixes(input_ontology, g_train)
    copy_prefixes(input_ontology, g_valid)

    decls_added_train = add_declarations_to(g_train, input_ontology)
    decls_added_valid = add_declarations_to(g_valid, input_ontology)

    # Helper to add an axiom to a destination graph, copying any bnode expressions
    def add_axiom(dst: Graph, subj: URIRef, obj: Union[URIRef, BNode]):
        if isinstance(obj, URIRef):
            dst.add((subj, RDFS.subClassOf, obj))
        else:
            mapping: Dict[BNode, BNode] = {}
            new_obj = _copy_bnode_closure(input_ontology, dst, obj, mapping)
            dst.add((subj, RDFS.subClassOf, new_obj))

    for s_node, o_node in train_pairs:
        add_axiom(g_train, s_node, o_node)
    for s_node, o_node in valid_pairs:
        add_axiom(g_valid, s_node, o_node)

    g_train += candidates_el

    # Ensure output directories exist
    train_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)

    g_train.serialize(destination=train_path.as_posix(), format="turtle")
    g_valid.serialize(destination=valid_path.as_posix(), format="turtle")

    logger.info(
        "Wrote train split: %s (decls=%d, axioms=%d)", train_path, decls_added_train, len(train_pairs)
    )
    logger.info(
        "Wrote valid split: %s (decls=%d, axioms=%d)", valid_path, decls_added_valid, len(valid_pairs)
    )


if __name__ == "__main__":
    main()
