from typing import List

_embedder_cache = None
_milvus_client_cache = None


def _get_embedder(settings: dict):
    global _embedder_cache
    if _embedder_cache is not None:
        return _embedder_cache
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers not installed") from e
    model_name = settings.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    _embedder_cache = SentenceTransformer(model_name)
    return _embedder_cache


def _get_milvus_client(settings: dict):
    global _milvus_client_cache
    if _milvus_client_cache is not None:
        return _milvus_client_cache
    try:
        from pymilvus import MilvusClient
    except Exception as e:
        raise RuntimeError("pymilvus not installed") from e
    uri = settings.get("vector_db_uri")
    _milvus_client_cache = MilvusClient(uri=uri)
    return _milvus_client_cache


def vector_top_k(query: str, elem_type: str, top_k: int, settings: dict) -> List[str]:
    embedder = _get_embedder(settings)
    client = _get_milvus_client(settings)
    q_vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    coll_class = settings.get("vector_collection_classes", "ref_classes")
    coll_prop = settings.get("vector_collection_properties", "ref_properties")

    def _search(coll_name: str):
        if not client.has_collection(coll_name):
            return []
        client.load_collection(coll_name)
        res = client.search(
            collection_name=coll_name,
            data=[q_vec],
            limit=max(top_k, 0),
            output_fields=["iri", "label", "definition"],
            search_params={"metric_type": "COSINE"},
        )
        hits = res[0] if res else []
        return [
            {"distance": h.get("distance", 0.0), "label": h['entity']['label'], "definition": h['entity']['definition'],
             "iri": h['entity']['iri']} for h in hits]

    t = (elem_type or "").strip().lower()
    if t == "class":
        scored = _search(coll_class)
    elif t == "property":
        scored = _search(coll_prop)
    else:
        # Unknown: search both and combine
        scored = (_search(coll_class) + _search(coll_prop))
    scored.sort(key=lambda x: x["distance"], reverse=True)
    return scored
