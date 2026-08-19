import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def search_topic(topic: str, max_results: int = 10) -> list[dict]:
    """Search Tavily for a research topic, return top results."""
    response = tavily_client.search(query=topic, max_results=max_results)
    return response.get("results", [])


def crawl_urls(urls: list[str]) -> list[dict]:
    """Fetch full content for a list of URLs using Tavily extract."""
    response = tavily_client.extract(urls=urls)
    return response.get("results", [])


def detect_format(content: str) -> str:
    """Very simple heuristic: markdown if it has markdown-style syntax, else plain text."""
    markdown_signals = ["##", "**", "- ", "```", "[", "](", "1. "]
    hits = sum(1 for signal in markdown_signals if signal in content)
    return "markdown" if hits >= 2 else "plain_text"