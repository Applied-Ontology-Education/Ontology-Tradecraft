import logging

import build_class_property_vector_db
import extract_ieo_definitions
import extract_reference_terms
import filter_candidates_hybrid
import generate_candidates_llm
import generate_phrase_differences_llm
import preprocess_definitions_llm
import robot_elk
import robot_merge
import split_train_valid
import train_mowl
from common.settings import build_settings
from util.logger_config import config

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
GENERATED_ROOT = PROJECT_ROOT / "generated"
REPORTS_ROOT = PROJECT_ROOT / "reports"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
settings = build_settings(PROJECT_ROOT, DATA_ROOT)

import os
os.environ["TQDM_DISABLE"] = "1"

def main():
    logger.info("=======================================")
    logger.info("Running all scripts")
    logger.info("=======================================")

    input_file = PROJECT_ROOT / "src/CommonCoreOntologiesMerged.ttl"
    output_file = settings["reference_ontology"]
    logger.info(f"Running ELK over {input_file}...")
    robot_elk.main(str(input_file), output_file)

    logger.info("Extracting reference terms...")
    extract_reference_terms.main()

    logger.info("Extracting definitions...")
    extract_ieo_definitions.main()

    logger.info("Building vector database...")
    build_class_property_vector_db.main()

    logger.info("Generating phrase differences...")
    generate_phrase_differences_llm.main()

    logger.info("Preprocessing definitions...")
    preprocess_definitions_llm.main()

    logger.info("Generating candidates ...")
    generate_candidates_llm.main()

    logger.info("Splitting train/valid ...")
    split_train_valid.main()

    logger.info("Training MOWL ...")
    train_mowl.main()

    logger.info("Filtering candidates ...")
    filter_candidates_hybrid.main()

    logger.info("Merging candidates ...")
    robot_merge.main()

    logger.info("Checking result with ELK ...")
    robot_elk.main()


if __name__ == "__main__":
    main()
