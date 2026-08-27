import asyncio
from src.agents.report_generation_agent import report_generation_agent
import time

fake_state = {
    "research_topic": "transformers",
    "research_questions": [],
    "web_documents": [],
    "ingestion_summary": {},
    "answers": [
        {
            "question": "What is self-attention?",
            "answer": "Self-attention is a mechanism that allows a model to assign different importance to different parts of the input.",
            "sources": ["https://fake2.com"],
            "retrieval_scores": {"mrr@10": 0.5, "ndcg@10": 0.62, "precision@5": 0.4, "recall@10": 1.0},
            "generation_scores": {"faithfulness": 1.0, "context_precision": 1.0, "answer_relevancy": 1.0},
        }
    ],
    "report_paths": {},
    "current_agent": "",
    "error": None,
    "retry_count": 0,
}

print("Starting agent 4...")
start = time.time()
result = asyncio.run(report_generation_agent(fake_state))
print(f"Agent 4 finished in {time.time() - start:.1f}s")
print("Report paths:", result["report_paths"])
print("Error:", result["error"])