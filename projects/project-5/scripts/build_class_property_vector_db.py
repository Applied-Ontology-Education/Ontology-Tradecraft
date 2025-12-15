import logging
from typing import List

import dotenv
import pandas as pd

from common.settings import build_settings
from util.logger_config import config

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)

dotenv.load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


def _load_entries(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize expected columns
    for col in ("iri", "label", "definition", "type"):
        if col not in df.columns:
            df[col] = ""
    # Clean
    df["label"] = df["label"].astype(str).str.strip()
    df["definition"] = df["definition"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.lower().replace({"": "unknown"})
    df["iri"] = df["iri"].astype(str).str.strip()
    # Filter out empty labels/definitions
    df = df[(df["label"] != "") & (df["definition"] != "")]
    return df


def _get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model


def _get_milvus_client(uri: str):
    from pymilvus import MilvusClient
    client = MilvusClient(uri=uri)
    return client


def _ensure_collection(client, name: str, dim: int):
    # Recreate collection to ensure schema matches current embedding dimension
    from pymilvus import DataType

    if client.has_collection(name):
        client.drop_collection(name)

    # Build explicit schema to include label/definition text fields
    schema = client.create_schema(
        auto_id=True,
        description="Reference terms embeddings",
        enable_dynamic_field=False,
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("iri", DataType.VARCHAR, max_length=512)
    schema.add_field("label", DataType.VARCHAR, max_length=512)
    schema.add_field("definition", DataType.VARCHAR, max_length=8192)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)

    # Prepare index params for vector field
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )


def _embed_texts(model, labels: List[str], defs: List[str]) -> List[List[float]]:
    texts = [f"{l} {d}" for l, d in zip(labels, defs)]
    embs = model.encode(texts, normalize_embeddings=True).tolist()
    return embs


def main():
    settings = build_settings(PROJECT_ROOT, DATA_ROOT)

    csv_path: Path = settings["bfo_cco_terms"]
    if not csv_path.exists():
        logger.error("Reference terms CSV not found: %s", csv_path)
        return

    df = _load_entries(csv_path)
    if df.empty:
        logger.warning("No entries to index from %s", csv_path)
        return

    embed_model_name = settings.get("embedding_model")
    logger.info("Loading embedding model: %s", embed_model_name)
    embedder = _get_embedder(embed_model_name)

    uri = settings.get("vector_db_uri")
    logger.info("Connecting to Milvus Lite at: %s", uri)
    client = _get_milvus_client(uri)

    dim = getattr(embedder, "get_sentence_embedding_dimension", lambda: 384)()
    if not isinstance(dim, int):
        dim = 384

    coll_class = settings.get("vector_collection_classes", "ref_classes")
    coll_prop = settings.get("vector_collection_properties", "ref_properties")

    # Prepare and insert classes
    for coll_name, type_name in ((coll_class, "class"), (coll_prop, "property")):
        sub = df[df["type"] == type_name]
        # Ensure collection exists (even if empty) for queries
        _ensure_collection(client, coll_name, dim)
        if sub.empty:
            logger.info("No %s entries to index.", type_name)
            client.load_collection(coll_name)
            continue
        iris = sub["iri"].tolist()
        labels = sub["label"].tolist()
        defs = sub["definition"].tolist()
        vectors = _embed_texts(embedder, labels, defs)
        # Convert to row-wise records for robust MilvusClient.insert consumption
        rows = [
            {"iri": i, "label": l, "definition": d, "vector": v}
            for i, l, d, v in zip(iris, labels, defs, vectors)
        ]
        logger.info("Inserting %d %s vectors into '%s'", len(labels), type_name, coll_name)
        client.insert(collection_name=coll_name, data=rows)
        client.load_collection(coll_name)

    logger.info("Vector database build complete.")


if __name__ == "__main__":
    main()
