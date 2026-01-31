from typing import Callable

from langchain.agents.middleware import (
    wrap_model_call, 
    ModelRequest, 
    ModelResponse,
    AgentMiddleware,
)
from langchain.tools import BaseTool


def dynamic_tools_factory(
    on_match_tools: list[BaseTool], 
    on_match_context_field: str,
    on_match_value: str,
) -> AgentMiddleware:

    @wrap_model_call
    def tools_changer(
        request: ModelRequest, 
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        field_value = getattr(request.runtime.context, on_match_context_field)
        
        if field_value == on_match_value:
            request = request.override(tools=on_match_tools)
        
        return handler(request)

    return tools_changer
