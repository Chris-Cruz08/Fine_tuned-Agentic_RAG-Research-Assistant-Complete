import src.config  # noqa: F401
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import ResearchState
from src.agents.web_intelligence_agent import web_intelligence_agent
from src.agents.ingestion_agent import ingestion_agent
from src.agents.rag_analysis_agent import rag_analysis_agent
from src.agents.report_generation_agent import report_generation_agent

MAX_RETRIES = 3


def route_after_web_intelligence(state: ResearchState) -> str:
    """If Agent 1 failed, retry it (up to MAX_RETRIES) or bail out entirely."""
    if state.get("error"):
        if state.get("retry_count", 0) < MAX_RETRIES:
            return "retry_web_intelligence"
        return "end_with_error"
    return "continue"


async def increment_retry(state: ResearchState) -> ResearchState:
    """Bump the retry counter and clear the error before trying the failing node again."""
    return {**state, "retry_count": state.get("retry_count", 0) + 1, "error": None}


def build_pipeline():
    graph = StateGraph(ResearchState)

    graph.add_node("web_intelligence", web_intelligence_agent)
    graph.add_node("retry_bump", increment_retry)
    graph.add_node("ingestion", ingestion_agent)
    graph.add_node("rag_analysis", rag_analysis_agent)
    graph.add_node("report_generation", report_generation_agent)

    graph.set_entry_point("web_intelligence")

    graph.add_conditional_edges(
        "web_intelligence",
        route_after_web_intelligence,
        {
            "continue": "ingestion",
            "retry_web_intelligence": "retry_bump",
            "end_with_error": END,
        },
    )
    graph.add_edge("retry_bump", "web_intelligence")

    graph.add_edge("ingestion", "rag_analysis")
    graph.add_edge("rag_analysis", "report_generation")
    graph.add_edge("report_generation", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)