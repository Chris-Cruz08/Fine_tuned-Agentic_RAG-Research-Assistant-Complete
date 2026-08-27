from src.evaluation.benchmark_runner import run_benchmark

results = run_benchmark()
print("Naive retrieval:   ", results["naive"])
print("Advanced retrieval:", results["advanced"])

from src.evaluation.benchmark_runner import run_generation_benchmark

print("\n--- Generation metrics ---")
gen_results = run_generation_benchmark()
for r in gen_results:
    print(f"\nQ: {r['question']}")
    print(f"Faithfulness: {r['faithfulness']}, Context Precision: {r['context_precision']}, Answer Relevancy: {r['answer_relevancy']}")

print("\n--- Debug: raw answer + context for low scores ---")
from src.agents.rag_analysis_agent import answer_question

for q in ["What are common challenges with RAG systems?", "How does chunking affect RAG performance?"]:
    data = answer_question(q)
    print(f"\nQ: {q}")
    print(f"Answer: {data['answer']}")
    print(f"Num context chunks: {len(data['top_chunks'])}")
    for i, c in enumerate(data["top_chunks"][:2]):
        print(f"  Context {i}: {c['text'][:150]}...")