import os
import sys
sys.path.append(os.getcwd())

import json
import uuid
from typing import Dict, Any

from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.langchain_syntax.tools.web_search import web_search
from src.langchain_syntax.llm.factory import get_mistral
from src.config import default_model_name


class BaseAgent:

    def __init__(
        self, 
        system_prompt: str,
        llm_name: str = default_model_name,
    ):
        # Один диалог на объект
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}  

        # Инициализация агента
        self.agent = create_agent(
            model=get_mistral(llm_name),
            tools=[web_search,],
            system_prompt=SystemMessage(content=system_prompt),
            checkpointer=InMemorySaver(),
        )


    def ask(self, query: str, in_dialog: bool = False) -> Dict[str, Any] | Any:
        question = HumanMessage(content=[{"type": "text", "text": query}])
        
        return self.agent.invoke(
            {"messages": [question]}, 
            config=self.config if in_dialog else None,
        )




if __name__ == "__main__":
    system_prompt = """
You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.
"""

    agent = BaseAgent(system_prompt)

    response = agent.ask("I have some leftover chicken and rice. What can I make?", in_dialog=True)

    print(response["messages"][-1].content)

    response_dict = {
        "messages": [
            {
                str(type(m)): m.model_dump()
            } for m in response["messages"]
        ]
    }

    with open(os.path.join(os.path.dirname(__file__), "base_agent_output.json"), "w", encoding="utf-8") as f:
        json.dump(response_dict, f, indent=4, ensure_ascii=False)
