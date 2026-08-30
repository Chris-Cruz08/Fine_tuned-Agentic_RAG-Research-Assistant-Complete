from src.agents.rag_analysis_agent import answer_question

# Full advanced pipeline
result_advanced = answer_question("What is self-attention?", use_query_expansion=True, use_reranking=True)
print("Advanced - queries used:", [result_advanced["rewritten_query"]] + result_advanced["expanded_queries"])
print("Advanced - answer:", result_advanced["answer"][:150])

print()

# Everything off - naive single query, no reranking
result_naive = answer_question("What is self-attention?", use_query_expansion=False, use_reranking=False)
print("Naive - queries used:", [result_naive["rewritten_query"]] + result_naive["expanded_queries"])
print("Naive - answer:", result_naive["answer"][:150])