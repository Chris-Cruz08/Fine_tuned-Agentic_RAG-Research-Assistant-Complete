import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.rag.embeddings import embedding_model
from src.rag.chunking import stable_id

COLLECTION_NAME = "research_chunks"
VECTOR_SIZE = 384  # bge-small-en-v1.5 output size

client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))


def ensure_collection():
    """Create the collection if it doesn't already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def store_chunks(chunks: list[dict]) -> dict:
    """Embed only child chunks (for retrieval) and store all chunks (parent+child) in Qdrant."""
    ensure_collection()

    child_chunks = [c for c in chunks if c["level"] == "child"]
    parent_chunks = [c for c in chunks if c["level"] == "parent"]

    texts_to_embed = [c["text"] for c in child_chunks]
    vectors = embedding_model.embed_documents(texts_to_embed) if texts_to_embed else []

    points = []
    for chunk, vector in zip(child_chunks, vectors):
        points.append(PointStruct(id=stable_id(chunk["id"]), vector=vector, payload=chunk))

    # Parent chunks stored with a zero-vector placeholder — retrieved by ID lookup, not similarity search
    for chunk in parent_chunks:
        points.append(PointStruct(
            id=stable_id(chunk["id"]),
            vector=[0.0] * VECTOR_SIZE,
            payload=chunk,
        ))

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    return {
        "parent_chunks_stored": len(parent_chunks),
        "child_chunks_stored": len(child_chunks),
        "total_points": len(points),
    }