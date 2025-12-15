import argparse
import io
import logging
import tempfile
from typing import Any

from rdflib import Graph

from common.robot import detect_robot, build_elk_robot_command, run
from common.vectorization import vector_top_k
from util.logger_config import config

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)

prefixes = """
@prefix cco: <https://www.commoncoreontologies.org/> .
@prefix obo: <http://purl.obolibrary.org/obo/> .
@prefix bfo: <http://purl.obolibrary.org/obo/> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xml: <http://www.w3.org/XML/1998/namespace> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
""".strip()


def elk_check_axioms(all_axioms: Graph, args: argparse.Namespace, axiom_graph: Graph, robot_dir: Path) -> int:
    # TODO add CCO graph
    status = 1
    try:
        test_graph = all_axioms + axiom_graph
        robot_cmd = detect_robot(args.robot, robot_dir)
        with tempfile.NamedTemporaryFile(suffix=".ttl") as in_path:
            with tempfile.NamedTemporaryFile(suffix=".ttl") as out_path:
                test_graph.serialize(format="turtle", destination=in_path.name)
                cmd = build_elk_robot_command(Path(in_path.name), Path(out_path.name), robot_cmd, args.max_mem)
                status = run(cmd)
    except Exception as e:
        logger.warning("ELK check failed: %s", e)
    return status


def parse_axiom(axiom) -> Graph:
    axiom_graph = None
    try:
        turtle = prefixes + "\n\n" + axiom
        print(turtle)
        axiom_graph = Graph()
        axiom_graph.parse(io.StringIO(turtle), format="turtle")
    except Exception as e:
        print(f"Failed to parse axiom: {axiom}")
    return axiom_graph


def add_candidates_from_other_words(other_words: list[str], ref_top_k: int, class_candidates: list[Any],
                                    property_candidates: list[Any],
                                    settings: dict) -> tuple[list[Any], list[Any]]:
    property_working_set = []
    class_working_set = []

    for word in other_words:
        property_vector_suggestions = vector_top_k(word, "property", 3 * ref_top_k, settings)
        for suggestion in property_vector_suggestions:
            property_working_set.append(suggestion)

        class_vector_suggestions = vector_top_k(word, "class", 3 * ref_top_k, settings)
        for suggestion in class_vector_suggestions:
            class_working_set.append(suggestion)

    property_working_set.sort(key=lambda x: x["distance"], reverse=True)
    property_working_set = property_working_set[:ref_top_k]

    class_working_set.sort(key=lambda x: x["distance"], reverse=True)
    class_working_set = class_working_set[:ref_top_k]

    for suggestion in property_working_set:
        property_candidates.append((suggestion["label"], suggestion["iri"]))

    for suggestion in class_working_set:
        class_candidates.append((suggestion["label"], suggestion["iri"]))



def add_candidates_from_labels(labels: set[str],
                               labels_to_iri: dict,
                               class_candidates: list[Any],
                               property_candidates: list[Any]):
    for label in labels:
        if label in labels_to_iri:
            if labels_to_iri[label]["type"] == "class":
                class_candidates.append((label, labels_to_iri[label]["iri"]))
            elif labels_to_iri[label]["type"] == "property":
                property_candidates.append((label, labels_to_iri[label]["iri"]))

    return class_candidates, property_candidates