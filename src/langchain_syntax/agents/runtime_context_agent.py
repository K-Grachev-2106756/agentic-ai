import os
import sys
sys.path.append(os.getcwd())

import random
import json
import uuid
from typing import Dict, Any, Tuple, Optional

from langchain.agents import create_agent, AgentState
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from src.langchain_syntax.llm.factory import get_mistral
from src.langchain_syntax.agents.tools.context import (
    context_getter_factory, 
    state_setter_factory, 
    state_getter_factory,
)
from src.config import default_model_name

from dataclasses import dataclass


class RuntimeAgent:

    def __init__(
        self, 
        system_prompt: Optional[str] = None,
        llm_name: str = default_model_name,
        tools: Optional[list[BaseTool]] = None,
        context_schema: Optional[Any] = None,
        state_schema: Optional[Any] = None,
    ):
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}  

        self.checkpointer = InMemorySaver()

        if context_schema is not None:
            self.context = context_schema()
        
        if system_prompt is None:
            system_prompt = (
                "You are an assistant with access to tools." 
                if tools else 
                "You are a helpful assistant"
            )
        
        self.agent = create_agent(
            model=get_mistral(model_name=llm_name),
            tools=tools if tools else None,
            system_prompt=SystemMessage(content=system_prompt),
            checkpointer=self.checkpointer,
            context_schema=context_schema,
            state_schema=state_schema,
        )


    def ask(self, query: str) -> Dict[str, Any] | Any:
        question = HumanMessage(content=[{"type": "text", "text": query}])
        
        return self.agent.invoke(
            {"messages": [question]}, 
            config=self.config,
            context=self.context if hasattr(self, "context") else None,
        )




if __name__ == "__main__":
    
    class PersonalInfo(AgentState):  # runtime state
        name: str
        age: int
        hobbies: list[str]

    
    @dataclass
    class ChatContext:  # constant runtime context
        taboo_topics: Tuple[str] = ("Politics", "Diseases", )


    get_context_tool = context_getter_factory(ChatContext)
    set_state_tool = state_setter_factory(PersonalInfo)
    get_state_tool = state_getter_factory(PersonalInfo)

    system_prompt = """
You are an assistant with access to tools.

Rules:
- If user provides personal information, you MUST store it using tool.
- You can read context using tool.
- You should rely on context and info you collected.
"""

    agent = RuntimeAgent(
        system_prompt=system_prompt,
        tools=[get_context_tool, set_state_tool, get_state_tool, ],
        context_schema=ChatContext,
        state_schema=PersonalInfo,
    )

    random_name = random.choice(["Peter", "Yan"])
    random_age = random.randint(10, 30)

    response = agent.ask(
        f"Hello. My name is {random_name} and I'm {random_age}. "
        "I'd like to practice small-talk. What topics should we not talk about? "
    )

    print(response["messages"][-1].content)

    assert agent.checkpointer.get(agent.config)["channel_values"]["name"] == random_name
    assert int(agent.checkpointer.get(agent.config)["channel_values"]["age"]) == random_age

    response_dict = {
        "messages": [
            {
                str(type(m)): m.model_dump()
            } for m in response["messages"]
        ]
    }

    with open(os.path.join(os.path.dirname(__file__), "runtime_context_agent_output.json"), "w", encoding="utf-8") as f:
        json.dump(response_dict, f, indent=4, ensure_ascii=False)
