from sentence_transformers import CrossEncoder

_reranker = None


def get_reranker():
    """Lazy-load the cross-encoder so it doesn't load until actually needed."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    return _reranker


def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """Re-score top chunks against the query using a cross-encoder, return top_n."""
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)

    scored_chunks = [{**chunk, "rerank_score": float(score)} for chunk, score in zip(chunks, scores)]
    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    return scored_chunks[:top_n]