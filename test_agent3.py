import asyncio
from src.agents.rag_analysis_agent import rag_analysis_agent

fake_state = {
    "research_topic": "transformers",
    "research_questions": ["What is self-attention?"],
    "web_documents": [],
    "ingestion_summary": {},
    "answers": [],
    "report_paths": {},
    "current_agent": "",
    "error": None,
    "retry_count": 0,
}

result = asyncio.run(rag_analysis_agent(fake_state))
for a in result["answers"]:
    print("Question:", a["question"])
    print("Answer:", a["answer"])
    print("Sources:", a["sources"])
    print("Top chunk parent_ids:", [c.get("parent_id") for c in a["top_chunks"]])