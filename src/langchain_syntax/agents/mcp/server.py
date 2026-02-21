import os
import sys
import logging
from typing import Dict, Any
sys.path.append(os.getcwd())

import aiohttp
import asyncio
from mcp.server.fastmcp import FastMCP

from src.langchain_syntax.tools.web_search import web_search_async
from src.logging_config import setup_logging


setup_logging()
logger = logging.getLogger(os.path.basename(__file__))

mcp = FastMCP("mcp_server")


@mcp.tool(name="web_search_mcp", description="Search the web for information")
async def web_search_mcp(query: str) -> str: # <--- Возвращаем str
    logger.info(f"USER QUERY:\n\n{query}")
    
    # Получаем результат поиска (предполагаем, что web_search_async.coroutine возвращает словарь или строку)
    search_result = await web_search_async.coroutine(query)
    search_result = search_result.get("results", [])
    if search_result:
        info_text = "\n".join(f"- {r['title']} ({r['url']}): {r['content']}" for r in search_result)
    else:
        info_text = "Nothing was found"
    
    return info_text


# Resources - provide access to langchain-ai repo files
@mcp.resource("github://langchain-ai/langchain-mcp-adapters/blob/main/README.md")
async def github_file() -> str:
    """Async resource for accessing langchain-ai/langchain-mcp-adapters/README.md"""
    url = "https://raw.githubusercontent.com/langchain-ai/langchain-mcp-adapters/main/README.md"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.text()
    except Exception as e:
        return f"Error: {str(e)}"


# Prompt template
@mcp.prompt()
async def prompt() -> str:
    """Analyze data from a langchain-ai repo file with comprehensive insights"""
    return """
You are a helpful assistant that answers user questions about LangChain, LangGraph and LangSmith.

You can use the following tools/resources to answer user questions:
- web_search_mcp: Search the web for information
- github_file: Access the langchain-ai repo files

If the user asks a question that is not related to LangChain, LangGraph or LangSmith, you should say "I'm sorry, I can only answer questions about LangChain, LangGraph and LangSmith."

You may try multiple tool and resource calls to answer the user's question.

You may also ask clarifying questions to the user to better understand their question.
"""




if __name__ == "__main__":
    mcp.run(transport="stdio")
    logger.info("mcp-server started")
