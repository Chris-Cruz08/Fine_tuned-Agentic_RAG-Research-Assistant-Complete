import asyncio
from src.agents.ingestion_agent import ingestion_agent

fake_state = {
    "research_topic": "test topic",
    "research_questions": [],
    "web_documents": [
        {"url": "https://fake1.com", "content": "# AI Research\n\nDeep learning is powerful. " * 30, "format": "markdown"},
        {"url": "https://fake2.com", "content": "Plain text about transformers and attention. " * 30, "format": "plain_text"},
    ],
    "ingestion_summary": {},
    "answers": [],
    "report_paths": {},
    "current_agent": "",
    "error": None,
    "retry_count": 0,
}

result = asyncio.run(ingestion_agent(fake_state))
print("Ingestion summary:", result["ingestion_summary"])
print("Error:", result["error"])