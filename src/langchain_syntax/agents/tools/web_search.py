from typing import Dict, Any

from langchain.tools import tool
from tavily import TavilyClient, AsyncTavilyClient

from src.config import tavily_api_key


_tavily = TavilyClient(tavily_api_key)
_tavily_async = AsyncTavilyClient(tavily_api_key)


@tool("web_search", description="Search the web for information")
def web_search(query: str) -> Dict[str, Any]:
    return _tavily.search(query)


@tool("web_search_async", description="Search the web for information")
async def web_search_async(query: str) -> Dict[str, Any]:
    return await _tavily_async.search(query)
