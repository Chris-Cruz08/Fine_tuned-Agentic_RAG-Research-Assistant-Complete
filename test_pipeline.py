import asyncio
from src.graph.pipeline import build_pipeline

async def main():
    pipeline = build_pipeline()

    initial_state = {
        "research_topic": "retrieval augmented generation techniques",
        "research_questions": [
            "What is retrieval augmented generation?",
            "What are common challenges with RAG systems?",
        ],
        "web_documents": [],
        "ingestion_summary": {},
        "answers": [],
        "report_paths": {},
        "current_agent": "",
        "error": None,
        "retry_count": 0,
    }

    config = {"configurable": {"thread_id": "test-run-1"}}
    result = await pipeline.ainvoke(initial_state, config=config)

    print("\n=== FINAL RESULT ===")
    print("Error:", result.get("error"))
    print("Ingestion summary:", result.get("ingestion_summary"))
    print("Number of answers:", len(result.get("answers", [])))
    print("Report paths:", result.get("report_paths"))

if __name__ == "__main__":
    asyncio.run(main())