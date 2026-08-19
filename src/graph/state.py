from typing import TypedDict, List, Dict, Any, Optional


class ResearchState(TypedDict):
    # Input
    research_topic: str
    research_questions: List[str]

    # Agent 1 output
    web_documents: List[Dict[str, Any]]  # url, content, format, raw_metadata

    # Agent 2 output
    ingestion_summary: Dict[str, Any]  # chunk counts, index stats

    # Agent 3 output
    answers: List[Dict[str, Any]]  # question, answer, sources, retrieval_scores, generation_scores

    # Agent 4 output
    report_paths: Dict[str, str]  # {"markdown": path, "pdf": path}

    # Control / error handling
    current_agent: str
    error: Optional[str]
    retry_count: int