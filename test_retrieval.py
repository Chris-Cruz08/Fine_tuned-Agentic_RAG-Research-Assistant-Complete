from src.rag.query_rewriting import rewrite_query
from src.rag.query_expansion import expand_query
from src.rag.retrieval import retrieve_children
from src.rag.rrf import reciprocal_rank_fusion

question = "What is self-attention in transformers?"

rewritten = rewrite_query(question)
variants = expand_query(rewritten)
all_queries = [rewritten] + variants

print("Queries used:")
for q in all_queries:
    print(" -", q)

ranked_lists = [retrieve_children(q, top_k=20) for q in all_queries]
fused = reciprocal_rank_fusion(ranked_lists)

print(f"\nTotal unique chunks after RRF: {len(fused)}")
print("Top 5 fused results:")
for chunk in fused[:5]:
    print(f"  score={chunk['rrf_score']:.4f} | {chunk['text'][:80]}...")

from src.rag.reranking import rerank

reranked = rerank(rewritten, fused[:20], top_n=5)
print("\nTop 5 after reranking:")
for chunk in reranked:
    print(f"  rerank_score={chunk['rerank_score']:.4f} | {chunk['text'][:80]}...")