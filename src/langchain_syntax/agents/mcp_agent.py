import os
import sys
import json
import logging
sys.path.append(os.getcwd())

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.messages import HumanMessage

from src.logging_config import setup_logging
from src.langchain_syntax.llm.factory import get_mistral
from src.langchain_syntax.agents.mcp.utils import sanitize_mcp_tool
from src.config import default_model_name


setup_logging()
logger = logging.getLogger(os.path.basename(__file__))

# start mcp-server
client = MultiServerMCPClient(
    {
        "local_server": {
                "transport": "stdio",
                "command": "python",
                "args": ["src/langchain_syntax/mcp/server.py"],
            }
    }
)


async def init_mcp():
    # get tools
    tools = await client.get_tools()
    sanitized_tools = [sanitize_mcp_tool(tool) for tool in tools]  # Обертка для mistralai
    logger.info("Loaded tools: %s", tools)

    # get resources
    resources = await client.get_resources("local_server")
    logger.info("Loaded resources: %s", resources)

    # get prompts
    prompt = await client.get_prompt("local_server", "prompt")
    prompt = prompt[0].content
    logger.info("Loaded prompt: %s", prompt)

    # agent initialization
    agent = create_agent(
        model=get_mistral(model_name=default_model_name),
        tools=sanitized_tools,
        system_prompt=prompt,
    )

    return agent


async def test_mcp():
    agent = await init_mcp()

    config = {"configurable": {"thread_id": "1"}}
    response = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Tell me about the langchain-mcp-adapters library")
            ],
        },
        config=config,
    )

    for m in response["messages"]:
        logger.info(f"{str(type(m))}: {m.content}")

    response_dict = {
        "messages": [
            {
                str(type(m)): m.model_dump()
            } for m in response["messages"]
        ]
    }

    with open(os.path.join(os.path.dirname(__file__), "mcp_agent_output.json"), "w", encoding="utf-8") as f:
        json.dump(response_dict, f, indent=4, ensure_ascii=False)




if __name__ == "__main__":
    asyncio.run(test_mcp())
