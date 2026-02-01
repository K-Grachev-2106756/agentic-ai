import os
import sys
sys.path.append(os.getcwd())

import json
from dataclasses import dataclass

from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.langchain_syntax.middleware.dynamic_models import dynamic_model_context_size_factory
from src.langchain_syntax.middleware.dynamic_prompt import dynamic_prompt_factory
from src.langchain_syntax.middleware.dynamic_tools import dynamic_tools_factory
from src.langchain_syntax.middleware.human_in_the_loop import (
    hitl_factory, 
    human_response_interpretation,
)
from src.langchain_syntax.llm.factory import get_mistral
from src.config import default_model_name, large_model_name


def init_email_agent():

    @dataclass
    class EmailContext:
        email_address: str = "julie@example.com"
        password: str = "password123"


    class AuthenticatedState(AgentState):
        authenticated: bool


    # Mocks
    @tool
    def check_inbox() -> str:
        """Check the inbox for recent emails"""
        return (
            "Hi Julie,\n"
            "I'm going to be in town next week and was wondering if we could grab a coffee?\n"
            "- best, Jane (jane@example.com)"
        )


    @tool
    def send_email(to: str, subject: str, body: str) -> str:
        """Send an response email"""
        return f"Email sent to {to} with subject {subject} and body {body}"


    @tool
    def authenticate(email: str, password: str, runtime: ToolRuntime) -> Command:
        """Authenticate the user with the given email and password"""
        if email == runtime.context.email_address and password == runtime.context.password:
            return Command(update={
                "authenticated": True, 
                "messages": [ToolMessage("Successfully authenticated", tool_call_id=runtime.tool_call_id)],
            })
        else:
            return Command(update={
                "authenticated": False,
                "messages": [ToolMessage("Authentication failed", tool_call_id=runtime.tool_call_id)],
            })

    # Динамические элементы агента
    dynamic_prompt = dynamic_prompt_factory(
        base_prompt="You are a helpful assistant that can authenticate users.",
        override_prompt="You are a helpful assistant that can check the inbox and send emails.",
        on_match_field="authenticated",
        on_match_value=True,
        field_storage="state",
    )

    dynamic_model = dynamic_model_context_size_factory(
        base_model_name=default_model_name, 
        override_model_name=large_model_name,
        context_size_threshold=10,
    )

    dynamic_tools = dynamic_tools_factory(
        base_tools=[authenticate],
        override_tools=[check_inbox, send_email],
        on_match_field="authenticated",
        on_match_value=True,
        field_storage="state",
    )

    # Human in the loop
    hitl = hitl_factory(
        tool_interrupt_mapping=dict(
            authenticate=False,
            check_inbox=False,
            send_email=True,
        )
    )

    agent = create_agent(
        model=get_mistral(default_model_name),
        tools=[authenticate, send_email, check_inbox],
        checkpointer=InMemorySaver(),
        state_schema=AuthenticatedState,
        context_schema=EmailContext,
        middleware=[
            dynamic_prompt,
            dynamic_model,
            dynamic_tools,
            hitl,
        ]
    )

    invoke_kwargs = dict(
        config={"configurable": {"thread_id": "1"}},
        context=EmailContext(),
    )

    return agent, invoke_kwargs




if __name__ == "__main__":
    agent, invoke_kwargs = init_email_agent()

    response = agent.invoke(
        {
            "messages": [HumanMessage(content="Check email please")]
        },
        **invoke_kwargs,
    )

    response = agent.invoke(
        {
            "messages": [HumanMessage(content="julie@example.com, password123")]
        },
        **invoke_kwargs,
    )

    response = agent.invoke(
        {
            "messages": [HumanMessage(content="Write I would glad to join her")]
        },
        **invoke_kwargs,
    )

    cmd = human_response_interpretation(
        llm=get_mistral(default_model_name), 
        human_msg="ok", 
        msg_history=response,
    )

    response = agent.invoke(cmd, **invoke_kwargs)

    response_dict = {
        "messages": [
            {
                str(type(m)): m.model_dump()
            } for m in response["messages"]
        ]
    }

    with open(os.path.join(os.path.dirname(__file__), "middleware_agent_output.json"), "w", encoding="utf-8") as f:
        json.dump(response_dict, f, indent=4, ensure_ascii=False)
