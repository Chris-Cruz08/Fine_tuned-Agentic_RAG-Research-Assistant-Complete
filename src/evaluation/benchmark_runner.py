import src.config  # noqa: F401
from src.evaluation.benchmark_data import BENCHMARK_QUESTIONS
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.rag.retrieval import retrieve_children
from src.rag.query_rewriting import rewrite_query
from src.rag.query_expansion import expand_query
from src.rag.rrf import reciprocal_rank_fusion
from src.rag.reranking import rerank
import mlflow
from datetime import datetime, timezone

def sanitize_metric_name(name: str) -> str:
    """MLflow metric names can't contain '@' — replace with '_at_'."""
    return name.replace("@", "_at_")


def label_relevant(chunks: list[dict], keywords: list[str]) -> set[str]:
    """Mark a chunk as relevant if its text contains any benchmark keyword (case-insensitive)."""
    relevant = set()
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        if any(kw.lower() in text_lower for kw in keywords):
            relevant.add(chunk["id"])
    return relevant


def run_naive_retrieval(question: str) -> list[dict]:
    """Naive baseline: single raw query, no rewriting/expansion/RRF/reranking."""
    return retrieve_children(question, top_k=20)


def run_advanced_retrieval(question: str) -> list[dict]:
    """Full advanced pipeline: rewrite -> expand -> retrieve x4 -> RRF -> rerank."""
    rewritten = rewrite_query(question)
    variants = expand_query(rewritten)
    all_queries = [rewritten] + variants
    ranked_lists = [retrieve_children(q, top_k=20) for q in all_queries]
    fused = reciprocal_rank_fusion(ranked_lists)
    return rerank(rewritten, fused[:20], top_n=20)


def run_benchmark() -> dict:
    """Run naive vs advanced retrieval across all benchmark questions, average the metrics."""
    naive_scores = []
    advanced_scores = []

    for item in BENCHMARK_QUESTIONS:
        question = item["question"]
        keywords = item["relevance_keywords"]

        naive_chunks = run_naive_retrieval(question)
        naive_relevant = label_relevant(naive_chunks, keywords)
        naive_ids = [c["id"] for c in naive_chunks]
        naive_scores.append(evaluate_retrieval(naive_ids, naive_relevant))

        advanced_chunks = run_advanced_retrieval(question)
        advanced_relevant = label_relevant(advanced_chunks, keywords)
        advanced_ids = [c["id"] for c in advanced_chunks]
        advanced_scores.append(evaluate_retrieval(advanced_ids, advanced_relevant))

    def average(scores: list[dict]) -> dict:
        keys = scores[0].keys()
        return {k: round(sum(s[k] for s in scores) / len(scores), 4) for k in keys}

    return {
        "naive": average(naive_scores),
        "advanced": average(advanced_scores),
    }

from src.agents.rag_analysis_agent import answer_question
from src.evaluation.generation_metrics import evaluate_generation


def run_generation_benchmark() -> list[dict]:
    """Run the full RAG answer pipeline + Ragas generation metrics for every benchmark question."""
    results = []
    for item in BENCHMARK_QUESTIONS:
        answer_data = answer_question(item["question"])
        contexts = [c["text"] for c in answer_data["top_chunks"]]

        scores = evaluate_generation(
            question=item["question"],
            answer=answer_data["answer"],
            contexts=contexts,
            reference=item["reference"],
        )
        results.append({
            "question": item["question"],
            "answer": answer_data["answer"],
            "faithfulness": scores.get("faithfulness"),
            "context_precision": scores.get("context_precision"),
            "answer_relevancy": scores.get("answer_relevancy"),
        })
    return results

def run_full_benchmark_with_logging():
    """Run both retrieval and generation benchmarks, logging everything to MLflow."""
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("ai-research-intelligence-system")

    run_name = f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        retrieval_results = run_benchmark()

        for k, v in retrieval_results["naive"].items():
            mlflow.log_metric(f"naive_{sanitize_metric_name(k)}", v)
        for k, v in retrieval_results["advanced"].items():
            mlflow.log_metric(f"advanced_{sanitize_metric_name(k)}", v)

        generation_results = run_generation_benchmark()
        for i, r in enumerate(generation_results):
            mlflow.log_metric(f"q{i}_faithfulness", r["faithfulness"] or 0)
            mlflow.log_metric(f"q{i}_context_precision", r["context_precision"] or 0)
            mlflow.log_metric(f"q{i}_answer_relevancy", r["answer_relevancy"] or 0)

        avg_faithfulness = sum(r["faithfulness"] or 0 for r in generation_results) / len(generation_results)
        avg_context_precision = sum(r["context_precision"] or 0 for r in generation_results) / len(generation_results)
        avg_answer_relevancy = sum(r["answer_relevancy"] or 0 for r in generation_results) / len(generation_results)
        mlflow.log_metric("avg_faithfulness", avg_faithfulness)
        mlflow.log_metric("avg_context_precision", avg_context_precision)
        mlflow.log_metric("avg_answer_relevancy", avg_answer_relevancy)

    return retrieval_results, generation_results