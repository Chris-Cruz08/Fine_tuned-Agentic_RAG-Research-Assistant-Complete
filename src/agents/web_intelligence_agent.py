from src.graph.state import ResearchState
from src.tools.tavily_tools import search_topic, crawl_urls, detect_format
from src.tools.mcp_tools import mcp_client


async def web_intelligence_agent(state: ResearchState) -> ResearchState:
    """Agent 1: search the topic, crawl top 5 URLs, save raw content via MCP."""
    topic = state["research_topic"]

    search_results = search_topic(topic, max_results=10)
    if not search_results:
        return {**state, "error": "No search results found", "current_agent": "web_intelligence_agent"}

    top_urls = [r["url"] for r in search_results[:5]]
    crawled = crawl_urls(top_urls)

    tools = await mcp_client.get_tools()
    save_tool = next(t for t in tools if t.name == "save_research_document")

    documents = []
    for item in crawled:
        content = item.get("raw_content", "") or ""
        fmt = detect_format(content)
        await save_tool.ainvoke({
            "url": item["url"],
            "content": content,
            "content_format": fmt,
        })
        documents.append({"url": item["url"], "content": content, "format": fmt})

    return {
        **state,
        "web_documents": documents,
        "current_agent": "web_intelligence_agent",
        "error": None,
    }