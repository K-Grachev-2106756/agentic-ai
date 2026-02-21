from typing import Any, Literal

from langchain.agents.middleware import (
    dynamic_prompt, 
    AgentMiddleware, 
    ModelRequest,
)

from .utils import request_reader


def dynamic_prompt_factory(
    base_prompt: str, 
    override_prompt: str, 
    on_match_field: str,
    on_match_value: Any,
    field_storage: Literal["state", "context"] = "context",
) -> AgentMiddleware:
    getter = request_reader(field_storage, on_match_field)
    
    @dynamic_prompt
    def prompt_changer(request: ModelRequest) -> str:
        field_value = getter(request)
                
        if field_value == on_match_value:
            return override_prompt
        
        return base_prompt

    return prompt_changer
