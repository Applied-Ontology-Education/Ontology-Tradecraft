import logging

import dotenv
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

from common.io import read_csv, write_df_to_csv, read_reference_entries
from common.ontology_utils import get_parent_by_label
from common.prompt_loading import load_markdown_prompt_templates
from common.settings import build_settings
from common.string_normalization import remove_snake_case, apply_label_casing, apply_single_quotes
from preprocess_definitions.definition_normalization import normalize_definition_prefix, create_class_definition_prompt, \
    create_property_definition_prompt, create_automatic_property_definition
from util.logger_config import config

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)

dotenv.load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


def main():
    logger.info("===================================================================")
    logger.info("Preprocessing definitions (LLM enrichment via LangChain + Ollama)")
    logger.info("===================================================================")

    # Load settings
    settings = build_settings(PROJECT_ROOT, DATA_ROOT)

    # Read input definitions CSV
    logger.info(f"Reading definitions CSV from: {settings['definitions_csv']}")
    definitions_df = read_csv(settings["definitions_csv"])

    # Configure LLM (Ollama must be running: `ollama serve` and model pulled)
    logger.info(f"Initializing LLM with model: {settings['model_name']}")

    # Load prompt config file
    logger.info(f"Loading prompt templates from: {settings['preprocess_definitions_llm_md']}")
    prompts = load_markdown_prompt_templates(settings["preprocess_definitions_llm_md"])
    # prompts, chains = build_prompts(llm, prompt_texts)

    # Load BFO/CCO reference entries
    ref_entries = read_reference_entries(settings)
    ref_labels = [element['label'] for element in ref_entries]

    # Load phrase differences
    phrase_diffs = pd.read_csv("data/phrase_differences.csv")
    phrase_diffs = phrase_diffs[["label", "difference"]]
    phrase_diffs_dict = phrase_diffs.set_index("label").to_dict()["difference"]

    # Set-up LLM communication
    llm = ChatOllama(
        model=settings["model_name"],  # or "gemma3n:latest" or a specific tag
        temperature=settings["temperature"],
    )

    # Perform the enrichment
    logger.info("Enriching definitions...")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("Building and querying reference context per row...")
    definitions_df['status'] = 'PENDING'
    success = 0
    config(logger)

    # Allow for retries
    for iter in range(10):
        out_rows = []
        for _, r in definitions_df.iterrows():
            iri = r["iri"]
            label = r["label"]
            status = r["status"]
            definition = r["definition"]
            type_name = r["type"]
            improved_definition = definition

            # Improve the definition if it has not already been improved
            if status != "OK":
                if type_name not in {"class", "property"}:
                    logger.error(f"Skipping {iri} ({label}) of type {type_name} as it is not a class or property.")
                    print(f"Skipping {iri} ({label}) of type {type_name} as it is not a class or property.")

                target_info = get_parent_by_label(label, type_name, settings["reference_ontology"])
                automatic_definition = create_automatic_property_definition(label, phrase_diffs_dict, target_info)

                if type_name == "class":
                    ollama_prompt = create_class_definition_prompt(iri,
                                                                   label,
                                                                   type_name,
                                                                   definition,
                                                                   automatic_definition,
                                                                   target_info,
                                                                   prompts,
                                                                   phrase_diffs_dict,
                                                                   ref_entries,
                                                                   ref_labels,
                                                                   settings)
                else:
                    ollama_prompt = create_property_definition_prompt(iri,
                                                                      label,
                                                                      type_name,
                                                                      definition,
                                                                      automatic_definition,
                                                                      target_info,
                                                                      prompts,
                                                                      phrase_diffs_dict,
                                                                      ref_entries,
                                                                      ref_labels,
                                                                      settings)

                logger.info(f"Querying LLM for improved definition of {iri} ({label})...")
                print(f"Querying LLM for improved definition of {iri} ({label})...")
                logger.info(ollama_prompt)
                print(ollama_prompt)
                chain = ollama_prompt | llm | JsonOutputParser()
                prompt_response = chain.invoke({})
                improved_definition: str = prompt_response['improved_definition']
                improved_definition = improved_definition.replace("if and only if", "iff")

                # Sanity check
                if type_name == "class":
                    if (("individual x" not in improved_definition)
                        or ("iff" not in improved_definition)
                        or ("'X'" in improved_definition)
                        or ("'Y'" in improved_definition)
                        or ("'Z'" in improved_definition)):
                        status = "NOK"
                    else:
                        status = "OK"
                        success += 1
                elif type_name == "property":
                    # Sanity check
                    if (("individual x" not in improved_definition)
                        or ("individual y" not in improved_definition)
                        or ("iff" not in improved_definition)
                        or ("'X'" in improved_definition)
                        or ("'Y'" in improved_definition)
                        or ("'C'" in improved_definition)
                        or ("'D'" in improved_definition)
                        or ("'Z'" in improved_definition)):
                        status = "NOK"
                    else:
                        status = "OK"
                        success += 1

                improved_definition = normalize_definition_prefix(improved_definition, type_name, automatic_definition)
                improved_definition = improved_definition.replace("individual x", "x")
                improved_definition = improved_definition.replace("individual y", "y")
                improved_definition = improved_definition.replace("[", "")
                improved_definition = improved_definition.replace("]", "")
                improved_definition = remove_snake_case(improved_definition)
                improved_definition = apply_label_casing(improved_definition, ref_labels)
                improved_definition = apply_single_quotes(improved_definition, ref_labels)

                logger.info("\n\n" + "#" * 80)
                logger.info(f"{status} (Done {success} out of {len(definitions_df)} rows.)")
                logger.info(f"Definition: {definition}")
                logger.info(f"Improved definition: {improved_definition}")
                logger.info("#" * 80 + "\n\n")

                print("\n\n" + "#" * 80)
                print(f"{status} (Done {success} out of {len(definitions_df)} rows.)")
                print(f"Definition: {definition}")
                print(f"Improved definition: {improved_definition}")
                print("#" * 80 + "\n\n")

                if status != "OK":
                    # Revert to original definition
                    improved_definition = definition

            out_rows.append({
                "iri": iri,
                "label": label,
                "type": r["type"],
                "status": status,
                "definition": improved_definition
            })

        definitions_df = pd.DataFrame(out_rows)
        definitions_df = definitions_df[["iri", "label", "type", "status", "definition"]]
        logger.info(f"Writing results CSV to: {settings['enriched_definitions']}")
        write_df_to_csv(definitions_df, settings["enriched_definitions"])
        logger.info("Checking if all definitions are improved.")
        if (definitions_df["status"] == "OK").all():
            break

    logger.info("Done.")


if __name__ == "__main__":
    main()
