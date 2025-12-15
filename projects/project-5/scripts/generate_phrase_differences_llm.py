import logging
from typing import Any, List

import dotenv
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from common.io import read_csv, write_df_to_csv
from common.ontology_utils import (
    get_parent_property_by_label,
    get_parent_class_by_label,
)
from common.prompt_loading import load_markdown_prompt_templates
from common.settings import build_settings
from util.logger_config import config

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)

dotenv.load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_ROOT / "phrase_differences.csv"


def main():
    logger.info("===================================================================")
    logger.info("Generating phrase differences (LLM via LangChain + Ollama)")
    logger.info("===================================================================")

    # Load settings
    settings = build_settings(PROJECT_ROOT, DATA_ROOT)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # Load prompt config file
    prompt_texts_file = settings['generate_phrase_differences_llm_md']
    logger.info(f"Loading prompt templates from: {prompt_texts_file}")
    prompt_texts = load_markdown_prompt_templates(prompt_texts_file)

    # Read input CSV (definitions)
    input_csv = settings.get("definitions_csv", DATA_ROOT / "definitions.csv")
    logger.info("Reading definitions from: %s", input_csv)
    df = read_csv(input_csv)

    # LLM
    logger.info("Initializing LLM with model: %s", settings["model_name"])
    llm = ChatOllama(model=settings["model_name"], temperature=settings["temperature"])

    # TTL path for parent lookup
    ttl_path = settings["reference_ontology"]
    if not ttl_path.exists():
        logger.warning("TTL file not found at %s; parent lookup will fail.", ttl_path)

    # Output rows
    df["status"] = "PENDING"
    df["difference"] = "None"
    df["raw"] = "None"
    df["parent_label"] = "None"
    for i in range(10):
        logger.info("Iteration %d", i)
        out_rows: List[dict[str, Any]] = []
        for _, r in df.iterrows():
            iri = str(r.get("iri", "") or "").strip()
            label = str(r.get("label", "") or "").strip()
            type_name = str(r.get("type", "") or "").strip().lower()
            status = str(r.get("status"))
            definition = str(r.get("definition") or "").strip()
            parent_label = str(r.get("parent_label"))
            difference = str(r.get("difference"))
            raw = str(r.get("raw"))

            # Only process classes and properties; skip others
            if type_name not in {"property", "class"}:
                continue

            if status == "OK":
                out_rows.append({
                    "iri": iri,
                    "label": label,
                    "type": type_name,
                    "parent_label": parent_label,
                    "difference": difference,
                    "status": status,
                    "raw": raw,
                })
                continue

            try:
                if type_name == "property":
                    info = get_parent_property_by_label(label, ttl_path)
                else:
                    info = get_parent_class_by_label(label, ttl_path)
                parent_label = info.get("parent_label")
            except Exception as e:
                logger.warning("Parent lookup failed for %s: %s", label, e)
                status = "OK"

            if not parent_label:
                status = "OK"
                difference = ""
            else:
                try:
                    # Build prompt messages and inject labels for X and Y
                    system_query = (prompt_texts[type_name]["system"]
                                    .replace("{X}", label)
                                    .replace("{Y}", parent_label)
                                    )

                    user_query = (prompt_texts[type_name]['user']
                                  .replace("{X}", label)
                                  .replace("{Y}", parent_label)
                                  )

                    ollama_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", system_query),
                            ("user", user_query),
                        ]
                    )

                    logger.info(user_query)
                    logger.info(system_query)

                    try:
                        # Invoke LLM
                        chain = ollama_prompt | llm | JsonOutputParser()
                        prompt_response = chain.invoke({})
                    except Exception as e:
                        logger.warning("LLM invocation failed for %s: %s", label, e)

                    # Parse LLM response
                    difference = prompt_response.get("description", "")
                    if difference.startswith("The difference between "):
                        status = "OK"
                    else:
                        status = "NOK"
                except Exception as e:
                    logger.warning("LLM invocation failed for %s: %s", label, e)
                    status = "NOK"
                    difference = ""

            logger.info(f"(status={status}) Label '{label}' has difference '{difference}'.")

            out_rows.append({
                "iri": iri,
                "label": label,
                "type": type_name,
                "parent_label": parent_label,
                "difference": difference,
                "status": status,
                "raw": raw,
            })

        df = pd.DataFrame(out_rows)
        if (df["status"] == "OK").all():
            break

    logger.info("Writing phrase differences to: %s", OUTPUT_FILE)
    write_df_to_csv(df, OUTPUT_FILE)
    logger.info("Done.")


if __name__ == "__main__":
    main()
