from typing import Callable, Any, Literal

from langchain.agents.middleware import (
    wrap_model_call, 
    ModelRequest, 
    ModelResponse,
    AgentMiddleware,
)

from src.langchain_syntax.llm.factory import get_mistral
from .utils import request_reader


def dynamic_model_context_size_factory(
    base_model_name: str,
    override_model_name: str,
    context_size_threshold: int = 10,
) -> AgentMiddleware:
    """Выбирает модель в зависимости от длины диалога"""

    @wrap_model_call
    def model_changer(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:

        new_model_name = (
            override_model_name 
            if len(request.messages) > context_size_threshold 
            else base_model_name
        )

        if request.model.model_name != new_model_name:
            request = request.override(model=get_mistral(new_model_name))

        return handler(request)

    return model_changer


def dynamic_model_factory(
    base_model_name: str,
    override_model_name: str,
    on_match_field: str,
    on_match_value: Any,
    field_storage: Literal["state", "context"] = "context",
) -> AgentMiddleware:
    getter = request_reader(field_storage, on_match_field)
    
    @wrap_model_call
    def model_changer(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:        
        field_value = getter(request)

        new_model_name = (
            override_model_name 
            if field_value == on_match_value
            else base_model_name
        )

        if request.model.model_name != new_model_name:
            request = request.override(model=get_mistral(new_model_name))

        return handler(request)

    return model_changer
