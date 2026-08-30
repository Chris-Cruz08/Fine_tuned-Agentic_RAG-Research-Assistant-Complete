from src.tools.qdrant_tools import client, COLLECTION_NAME
from src.rag.embeddings import embedding_model
from src.rag.chunking import stable_id
from langsmith import traceable

@traceable(name="retrieve_children", run_type="retriever")
def retrieve_children(query: str, top_k: int = 20) -> list[dict]:
    """Retrieve top_k child chunks from Qdrant for a single query, ranked by similarity."""
    query_vector = embedding_model.embed_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter={"must": [{"key": "level", "match": {"value": "child"}}]},
    )

    return [
        {"id": point.id, "score": point.score, **point.payload}
        for point in results.points
    ]

def get_parent_chunks(reranked_children: list[dict]) -> list[dict]:
    """Given reranked child chunks, fetch their unique parent chunks by ID."""
    parent_ids = list({chunk["parent_id"] for chunk in reranked_children if chunk.get("parent_id")})
    if not parent_ids:
        return []

    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[stable_id(pid) for pid in parent_ids],
        with_payload=True,
    )
    return [point.payload for point in points]