from typing import Any, Annotated, Dict
from dataclasses import is_dataclass

from pydantic import Field
from langchain_core.tools.base import BaseTool
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command


def context_getter_factory(context_dataclass: Any) -> BaseTool:
    """
    Универсальная функция для создания runtime context getter.
    Args:
        context - ваш dataclass
    """
    context = context_dataclass()
    available_fields = list(context.__dataclass_fields__.keys())

    @tool(
        name_or_callable=f"{context_dataclass.__name__}_getter",
        description=f"""Get context about {"/".join(available_fields)}""",
    )
    def getter(
        field: Annotated[
            str, 
            Field(description=f"One of: {', '.join(available_fields)}"),
        ], 
        runtime: ToolRuntime,
    ):
        if field not in available_fields:
            raise ValueError(f"Invalid field '{field}', must be one of {available_fields}")
        
        return getattr(runtime.context, field)

    return getter


def state_getter_factory(state_dataclass: Any) -> BaseTool:
    """
    Универсальная функция для создания runtime state getter.
    Args:
        state_dataclass - Dict-like (AgentState)
    """
    # AgentState — dict-like объект
    state = state_dataclass()
    available_fields = list(state.keys())

    @tool(
        name_or_callable=f"{state_dataclass.__name__}_getter",
        description=f"""Get info about {"/".join(available_fields)} from the state.""",
    )
    def getter(
        field: Annotated[
            str,
            Field(description=f"One of: {', '.join(available_fields)}"),
        ],
        runtime: ToolRuntime,
    ):
        if field not in runtime.state:
            raise ValueError(
                f"Invalid field '{field}', must be one of {available_fields}"
            )

        return runtime.state.get(field)

    return getter


def state_setter_factory(state_dataclass: Any) -> BaseTool:
    """
    Универсальная функция для создания runtime state setter.
    Args:
        context - ваш Dict(langchain.agents.AgentState)
    """
    state = state_dataclass()
    available_fields = list(state.keys())

    @tool(
        name_or_callable=f"{state_dataclass.__name__}_setter",
        description=f"""Update info about {"/".join(available_fields)} in the state once user has revealed it.""",
    )
    def setter(
        field: Annotated[
            str, 
            Field(description=f"One of: {', '.join(available_fields)}"),
        ], 
        value: Any, 
        runtime: ToolRuntime,
    ) -> Command:
        return Command(
            update={
                field: value,
                "messages": [
                    ToolMessage(f"Successfully updated {field}", tool_call_id=runtime.tool_call_id),
                ],
            },
        )

    return setter
