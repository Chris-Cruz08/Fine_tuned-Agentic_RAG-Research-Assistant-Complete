import os
import json
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("research-file-server")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
RAW_DIR = os.path.join(BASE_DIR, "raw_documents")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


@mcp.tool()
def save_research_document(url: str, content: str, content_format: str) -> str:
    """Save a scraped research document to local disk with metadata.

    Args:
        url: source URL of the document
        content: the raw text/markdown content
        content_format: detected format, e.g. "markdown" or "plain_text"
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in url)[:60]
    filename = f"{timestamp}_{safe_name}.json"
    filepath = os.path.join(RAW_DIR, filename)

    payload = {
        "url": url,
        "content": content,
        "format": content_format,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return f"Saved document from {url} to {filepath}"


@mcp.tool()
def save_report(markdown_content: str, filename_prefix: str = "research_report") -> str:
    """Save the final research report as a Markdown file.

    Args:
        markdown_content: the full report content in markdown
        filename_prefix: prefix for the output filename
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(REPORTS_DIR, f"{filename_prefix}_{timestamp}.md")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return f"Report saved to {filepath}"


if __name__ == "__main__":
    mcp.run(transport="stdio")