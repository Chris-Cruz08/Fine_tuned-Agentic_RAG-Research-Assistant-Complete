from src.evaluation.retrieval_metrics import evaluate_retrieval

# Pretend these are ranked chunk IDs returned by search, best match first
ranked_ids = ["chunk_5", "chunk_2", "chunk_9", "chunk_1", "chunk_7"]

# Pretend we know chunk_2 and chunk_7 are the actually-correct answers for this question
relevant_ids = {"chunk_2", "chunk_7"}

scores = evaluate_retrieval(ranked_ids, relevant_ids)
print("Retrieval metrics:", scores)