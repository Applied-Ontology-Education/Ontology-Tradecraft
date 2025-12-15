import argparse
import logging
import os
from pathlib import Path
from typing import List

import nltk
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from nltk.corpus import stopwords
from owlready2 import sync_reasoner
from pydantic import BaseModel
from rdflib import Graph
from rdflib import URIRef

from check_domain import _load_ontology_any, check_domain_range
from common.io import load_entries, read_csv
from common.ontology_utils import get_turtle_snippet_by_uri_ref, get_parent_by_label
from common.prompt_loading import load_markdown_prompt_templates
from common.settings import build_settings
from common.string_normalization import collect_labels, filter_labels, collect_words, filter_stopwords
from generate_candidates.subroutines import elk_check_axioms, parse_axiom, add_candidates_from_other_words, \
    add_candidates_from_labels
from util.logger_config import config

PROJECT_ROOT = Path(__file__).parent.parent
ROBOT_DIR = PROJECT_ROOT / "robot"
DATA_ROOT = PROJECT_ROOT / "data"
path = PROJECT_ROOT / Path("prompts/generate_candidates_llm.md")

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)

nltk.download('stopwords')


class LlmResponse(BaseModel):
    candidate_axioms: List[str]
    reasoning: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate candidate axioms for IEO using Ollama."
        )
    )
    p.add_argument(
        "--robot",
        default=None,
        help="Path to robot executable or robot.jar (if not provided, auto-detect)",
    )
    p.add_argument(
        "--max-mem",
        default=os.getenv("ROBOT_JAVA_MAX_MEM", "6g"),
        help="Max Java heap for ROBOT when using robot.jar (default: 6g)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    settings = build_settings(PROJECT_ROOT, DATA_ROOT)

    stop_words = set(stopwords.words('english'))

    # Get reference ontology
    reference_ontology = Graph()
    reference_ontology.parse(settings["reference_ontology"], format='turtle')

    # Read enriched definitions CSV
    logger.info(f"Reading definitions CSV from: {settings['enriched_definitions']}")
    df = read_csv(settings["enriched_definitions"])
    df.fillna("", inplace=True)

    # Get BFO and CCO reference terms
    csv_path: Path = settings["bfo_cco_terms"]
    if not csv_path.exists():
        logger.error("Reference terms CSV not found: %s", csv_path)
        return

    reference_terms_df = load_entries(csv_path)
    reference_terms_df.fillna("", inplace=True)
    if reference_terms_df.empty:
        logger.warning("No entries to index from %s", csv_path)
        return

    # Create mapping from class and property labels to IRIs
    labels_to_iri = {row['label']: {"iri": row["iri"], "type": row["type"]} for _, row in reference_terms_df.iterrows()}
    iri_to_type = {row["iri"]: row["type"] for _, row in reference_terms_df.iterrows()}

    # Phrase difference
    phrase_differences_path: Path = settings["phrase_differences"]
    if not csv_path.exists():
        logger.error("Phrase difference CSV not found: %s", phrase_differences_path)
        return

    phrase_differences_df = pd.read_csv(phrase_differences_path, header=0)
    phrase_differences_df.fillna("", inplace=True)
    if reference_terms_df.empty:
        logger.warning("No entries to index from %s", phrase_differences_path)
        return

    # Create mapping from class and property labels to IRIs
    labels_to_phrase_difference = {row['label']: row["difference"] for _, row in phrase_differences_df.iterrows()}

    # BFO and CCO reference ontologies
    ttl_path: Path = settings["reference_ontology"]
    if not ttl_path.exists():
        logger.error("Reference ontology not found: %s", ttl_path)
        return

    reference_ontology = Graph()
    reference_ontology.parse(ttl_path.as_posix(), format='turtle')
    if len(reference_ontology.all_nodes()) == 0:
        logger.warning("No entries in reference ontology %s", ttl_path)
        return

    # Prepare OWLready2 domain and range checks
    onto_path = Path(ttl_path) if ttl_path else None
    if onto_path is None or not onto_path.exists():
        raise FileNotFoundError(
            f"Ontology not found: {onto_path}. Provide --ontology or set reference_ontology in settings."
        )

    onto = _load_ontology_any(onto_path)

    # Classify once
    with onto:
        # Owlready2 typically uses HermiT-like reasoner through Java bridge
        sync_reasoner()

    # Set-up LLM communication
    llm = ChatOllama(
        model=settings["model_name"],  # or "gemma3n:latest" or a specific tag
        temperature=settings["temperature"],
    )

    # Load prompt config file
    logger.info(f"Loading prompt templates from: {settings['generate_candidates_llm_md']}")
    prompt_texts = load_markdown_prompt_templates(path)

    # Perform the enrichment
    logger.info("Generating axioms...")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    settings["generated_root"].mkdir(parents=True, exist_ok=True)
    logger.info("Building and querying prompt per row...")
    df['status'] = 'PENDING'
    all_axioms = Graph()
    success = 0
    invalid_axioms = {}

    # Allow for retries
    for iter in range(10):
        out_rows = []
        for _, r in df.iterrows():
            iri = r["iri"]
            label = r["label"]
            status = r["status"]
            definition = r["definition"]
            type_name = r["type"]

            if type_name in {"property"}:
                logger.info(f"Skipping {iri} ({label}) of type {type_name}.")
                status = "OK"

            if status != 'OK':
                owl_definition = get_turtle_snippet_by_uri_ref(URIRef(iri), type_name,
                                                               PROJECT_ROOT / "src/CommonCoreOntologiesMerged.ttl")

                target_info = get_parent_by_label(label, type_name, settings["reference_ontology"])

                # Create mapping candidates
                ref_top_k = 25  # settings.get("ref_top_k")
                property_candidates = []
                class_candidates = []

                # Collect labels and words
                work_definition = definition
                candidate_labels = collect_labels(work_definition, reference_terms_df["label"].tolist())
                work_definition = filter_labels(work_definition, candidate_labels)
                work_definition = work_definition + labels_to_phrase_difference[label]
                other_words = collect_words(work_definition)
                other_words = filter_stopwords(other_words, stop_words)

                # Collect directly recognized class and property labels
                labels = collect_labels(definition, reference_terms_df["label"].tolist())
                add_candidates_from_labels(labels, labels_to_iri, class_candidates, property_candidates)

                # Do a vector search for other candidates
                add_candidates_from_other_words(other_words, ref_top_k,
                                                class_candidates,
                                                property_candidates,
                                                settings)

                # Filter out parent candidates, too many false positives with this
                # property_candidates = [ candidate for candidate in property_candidates if candidate[1] != target_info['parent_iri']]
                # class_candidates = [ candidate for candidate in class_candidates if candidate[1] != target_info['parent_iri'] and candidate[1] != iri]

                # Filter property candidates that have a compatible domain
                filtered_property_candidates = [
                    candidate
                    for candidate in property_candidates
                    if check_domain_range(onto, iri, candidate[1], "domain")
                ]

                # Skip it
                if len(filtered_property_candidates) == 0:
                    status = "OK"
                    out_rows.append({
                        "iri": iri,
                        "label": label,
                        "type": type_name,
                        "status": status,
                        "definition": definition
                    })
                    continue

                # Consolidate property candidates
                seen_properties = set()
                final_properties = []
                for candidate in filtered_property_candidates:
                    if not candidate[1] in seen_properties:
                        final_properties.append(candidate)
                        seen_properties.add(candidate[1])

                final_properties = final_properties[:ref_top_k]
                class_candidates = class_candidates[:ref_top_k]

                # Create mapping sections
                properties_section = ""
                for candidate in final_properties:
                    properties_section += f"- '{candidate[0]}': {candidate[1]}\n"

                classes_section = ""
                for candidate in set(class_candidates):
                    classes_section += f"-'{candidate[0]}': {candidate[1]}\n"

                # invalid_axiom_text = "No invalid axioms have been identified for this definition."
                # if iri in invalid_axioms:
                #     invalid_axiom_text = "Invalid axioms have been identified for this definition:\n" + "\n".join(
                #         [f"- {', '.join(values)}" for axiom, values in invalid_axioms[iri].items()])

                # Read and prepare the prompt
                system_query = (prompt_texts[type_name]["system"]
                                .replace("{target_definition}", definition)
                                .replace("{target_uri_ref}", iri)
                                .replace("{owl_definition}", owl_definition)
                                .replace("{properties_section}", properties_section)
                                .replace("{classes_section}", classes_section)
                                # .replace("{invalid_axiom_text}", invalid_axiom_text)
                                )

                user_query = (prompt_texts[type_name]['user']
                              .replace("{target_definition}", definition)
                              .replace("{target_uri_ref}", iri)
                              .replace("{owl_definition}", owl_definition)
                              .replace("{properties_section}", properties_section)
                              .replace("{classes_section}", classes_section)
                              # .replace("{invalid_axiom_text}", invalid_axiom_text)
                              )

                logger.info(system_query)
                print(system_query)
                logger.info(user_query)
                print(user_query)

                ollama_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_query),
                        ("user", user_query),
                    ]
                )

                # Run the query
                logger.info(f"Querying LLM for axiom candidates for {iri} ({label})...")
                logger.info(ollama_prompt)
                try:
                    chain = ollama_prompt | llm | JsonOutputParser()
                    prompt_response = chain.invoke({})

                    # Parse LLM response
                    if "candidate_axioms" in prompt_response:
                        axiom_responses: List[str] = prompt_response['candidate_axioms']
                    else:
                        axiom_responses = []

                    # Accept empty responses
                    if len(axiom_responses) == 0:
                        success += 1
                        status = "OK"

                    if "reasoning" in prompt_response:
                        logger.info('.\n'.join(prompt_response['reasoning'].split('.')))
                        print('.\n'.join(prompt_response['reasoning'].split('.')))

                except Exception as e:
                    logger.warning("LLM response parsing failed for %s: %s", label, e)

                    # Do not accept malformed responses.
                    axiom_responses = []
                    status = "NOK"

                # Parse axioms
                for axiom in axiom_responses:
                    class_uri = axiom['class_uri']
                    property_uri = axiom['verb_phrase_uri']
                    filler_uri = axiom['noun_phrase_uri']
                    # Basic soundness check
                    if (class_uri not in iri_to_type
                        or property_uri not in iri_to_type
                        or filler_uri not in iri_to_type
                        or not iri_to_type[class_uri] == "class"
                        or not iri_to_type[property_uri] == "property"
                        or not iri_to_type[filler_uri] == "class"):
                        print("Failed soundness check")
                        status = "NOK"
                    else:
                        # Solid range check on filler
                        # if not check_domain_range(onto, property_uri, filler_uri, "range"):
                        #     values = invalid_axioms.get(iri, [])
                        #     values.append((iri, property_uri, filler_uri))
                        #     invalid_axioms[iri] = values
                        #     status = "NOK"
                        # else:
                        axiom_snippet = """
                        <{class_uri}>
                            rdfs:subClassOf [   a                  owl:Restriction ;
                                                owl:onProperty     <{verb_phrase_uri}> ;
                                                owl:someValuesFrom <{noun_phrase_uri}> ] .
                        """.strip()
                        axiom_snippet = axiom_snippet.replace("{class_uri}", class_uri)
                        axiom_snippet = axiom_snippet.replace("{verb_phrase_uri}", property_uri)
                        axiom_snippet = axiom_snippet.replace("{noun_phrase_uri}", filler_uri)
                        axiom_graph = parse_axiom(axiom_snippet)
                        if axiom_graph is not None:
                            elk_status = elk_check_axioms(reference_ontology, args, axiom_graph, ROBOT_DIR)
                            if elk_status != 0:
                                logger.info("Axiom failed ELK reasoner: %s")
                                logger.info("Axiom: %s", axiom)
                            else:
                                all_axioms += axiom_graph
                                status = "OK"
                        else:
                            status = "NOK"

                if status == "OK":
                    success += 1
                    logger.info(f"Success ({success} / {len(df)})")
                    print(f"Success ({success} / {len(df)})")
                else:
                    logger.info(f"Refused ({success} / {len(df)})")
                    print(f"Refused ({success} / {len(df)})")

                # Write results to file
                all_axioms.serialize(format="turtle", destination=settings["candidates_el"])

            out_rows.append({
                "iri": iri,
                "label": label,
                "type": type_name,
                "status": status,
                "definition": definition
            })

        # Check if all definitions have been processed successfully.
        df = pd.DataFrame(out_rows)
        if (df["status"] == "OK").all():
            break


if __name__ == "__main__":
    main()
