from typing import Callable, Any, Literal

from langchain.agents.middleware import (
    wrap_model_call, 
    ModelRequest, 
    ModelResponse,
    AgentMiddleware,
)
from langchain.tools import BaseTool

from .utils import request_reader


def dynamic_tools_factory(
    base_tools: list[BaseTool],
    override_tools: list[BaseTool], 
    on_match_field: str,
    on_match_value: Any,
    field_storage: Literal["state", "context"] = "context",
) -> AgentMiddleware:
    getter = request_reader(field_storage, on_match_field)
    
    @wrap_model_call
    def tools_changer(
        request: ModelRequest, 
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        field_value = getter(request)
        
        request = request.override(
            tools=override_tools if field_value == on_match_value else base_tools
        )

        return handler(request)

    return tools_changer
