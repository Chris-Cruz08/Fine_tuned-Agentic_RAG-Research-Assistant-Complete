from langsmith import traceable

@traceable(name="reciprocal_rank_fusion", run_type="tool")
def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Merge multiple ranked lists of chunks into one, using Reciprocal Rank Fusion.
    Each chunk's RRF score = sum over lists of 1 / (k + rank), rank starting at 1.
    """
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            chunk_id = chunk["id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_lookup[chunk_id] = chunk

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {**chunk_lookup[chunk_id], "rrf_score": score}
        for chunk_id, score in fused
    ]