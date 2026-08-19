import os
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER_PATH = os.path.join(os.path.dirname(__file__), "..", "mcp_server", "server.py")

mcp_client = MultiServerMCPClient(
    {
        "research_files": {
            "command": "uv",
            "args": ["run", "python", SERVER_PATH],
            "transport": "stdio",
        }
    }
)


async def get_mcp_tools():
    """Fetch MCP tools as LangChain-compatible tools for use in agents."""
    return await mcp_client.get_tools()