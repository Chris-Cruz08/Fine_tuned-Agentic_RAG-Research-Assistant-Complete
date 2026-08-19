from src.graph.state import ResearchState
from src.rag.chunking import chunk_document
from src.tools.qdrant_tools import store_chunks


async def ingestion_agent(state: ResearchState) -> ResearchState:
    """Agent 2: chunk all documents from Agent 1 and store them in Qdrant."""
    documents = state.get("web_documents", [])

    if not documents:
        return {**state, "error": "No documents to ingest", "current_agent": "ingestion_agent"}

    total_parents = 0
    total_children = 0

    for doc in documents:
        chunks = chunk_document(
            content=doc["content"],
            content_format=doc["format"],
            source_url=doc["url"],
        )
        result = store_chunks(chunks)
        total_parents += result["parent_chunks_stored"]
        total_children += result["child_chunks_stored"]

    summary = {
        "documents_processed": len(documents),
        "total_parent_chunks": total_parents,
        "total_child_chunks": total_children,
    }

    return {
        **state,
        "ingestion_summary": summary,
        "current_agent": "ingestion_agent",
        "error": None,
    }