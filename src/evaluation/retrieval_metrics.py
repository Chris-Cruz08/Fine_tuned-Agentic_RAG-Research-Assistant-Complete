import math


def mrr_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant result in top k, else 0."""
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain. Relevance is binary (1 if relevant, else 0)."""
    dcg = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        relevance = 1.0 if chunk_id in relevant_ids else 0.0
        dcg += relevance / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    """What fraction of the top k retrieved results are actually relevant."""
    if k == 0:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """What fraction of all relevant chunks were found in the top k."""
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def evaluate_retrieval(ranked_ids: list[str], relevant_ids: set[str]) -> dict:
    """Run all 4 retrieval metrics at once."""
    return {
        "mrr@10": round(mrr_at_k(ranked_ids, relevant_ids, k=10), 4),
        "ndcg@10": round(ndcg_at_k(ranked_ids, relevant_ids, k=10), 4),
        "precision@5": round(precision_at_k(ranked_ids, relevant_ids, k=5), 4),
        "recall@10": round(recall_at_k(ranked_ids, relevant_ids, k=10), 4),
    }