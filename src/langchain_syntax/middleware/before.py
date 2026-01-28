from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import before_agent
from langchain.messages import RemoveMessage, ToolMessage
from langgraph.runtime import Runtime


@before_agent
def trim_tool_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Remove all tool messages from the state"""
    return {
        "messages": [
            RemoveMessage(id=m.id) for m in state["messages"] if isinstance(m, ToolMessage)
        ]
    }
