import logging
import os

from util.logger_config import config

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)


def build_settings(PROJECT_ROOT, DATA_ROOT) -> dict:
    """Collect all runtime settings into a single dictionary.

    Values are derived from environment variables with sensible defaults
    and project-relative paths.
    """

    GENERATED_ROOT = PROJECT_ROOT / "generated"

    # File paths
    definitions_csv = DATA_ROOT / "definitions.csv"
    bfo_cco_terms = DATA_ROOT / "bfo_cco_terms.csv"
    enriched_definitions_file = DATA_ROOT / "definitions_enriched.csv"
    phrase_differences = DATA_ROOT / "phrase_differences.csv"
    candidates_el = GENERATED_ROOT / "candidates_el.ttl"

    # Environment-driven parameters
    ref_top_k = int(os.getenv("REF_TOP_K", "5"))
    reference_ontology = Path(os.getenv("REFERENCE_ONTOLOGY", PROJECT_ROOT / "src" / "CommonCoreOntologiesMergedReasoned.ttl"))

    # Ollama
    model_name = os.getenv("OLLAMA_MODEL", "gemma3n")
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
    generate_phrase_differences_llm_md = Path("GENERATE_PHRASE_DIFFERENCES_LLM_MD", PROJECT_ROOT / "prompts" / "generate_phrase_differences_llm.md")
    generate_candidates_llm_md = Path(
        os.getenv("GENERATE_CANDIDATESs_LLM", PROJECT_ROOT / "prompts" / "generate_candidates_llm_md.md"))
    preprocess_definitions_llm_md = Path(
        os.getenv("PREPROCESS_DEFINITIONS_LLM_MD", PROJECT_ROOT / "prompts" / "preprocess_definitions_llm.md"))

    # Vector DB / Embeddings (optional; used when REFERENCE_MODE=vector)
    vector_db_uri = os.getenv("VECTOR_DB_URI", str(DATA_ROOT / "milvus.db"))
    vector_collection_classes = os.getenv("VECTOR_COLLECTION_CLASSES", "ref_classes")
    vector_collection_properties = os.getenv("VECTOR_COLLECTION_PROPERTIES", "ref_properties")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Filter candidates based on similarity to reference ontology
    tau = 0.7

    settings = {
        "project_root": PROJECT_ROOT,
        "data_root": DATA_ROOT,
        "generated_root": GENERATED_ROOT,
        "definitions_csv": definitions_csv,
        "candidates_el": candidates_el,
        "bfo_cco_terms": bfo_cco_terms,
        "enriched_definitions": enriched_definitions_file,
        "ref_top_k": ref_top_k,
        "preprocess_definitions_llm_md": preprocess_definitions_llm_md,
        "generate_phrase_differences_llm_md": generate_phrase_differences_llm_md,
        "generate_candidates_llm_md": generate_candidates_llm_md,
        "model_name": model_name,
        "temperature": temperature,
        "reference_ontology": reference_ontology,
        "phrase_differences": phrase_differences,
        # Vector search
        "vector_db_uri": vector_db_uri,
        "vector_collection_classes": vector_collection_classes,
        "vector_collection_properties": vector_collection_properties,
        "embedding_model": embedding_model,
        "tau": tau
    }

    logger.info("Settings:")
    for k, v in settings.items():
        logger.info("\t%s: %s", k, v)

    return settings
