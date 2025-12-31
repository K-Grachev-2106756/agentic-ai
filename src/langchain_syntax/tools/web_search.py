from typing import Dict, Any

from langchain.tools import tool
from tavily import TavilyClient

from src.config import tavily_api_key


_tavily = TavilyClient(tavily_api_key)


@tool("web_search", description="Search the web for information")
def web_search(query: str) -> Dict[str, Any]:
    return _tavily.search(query)
