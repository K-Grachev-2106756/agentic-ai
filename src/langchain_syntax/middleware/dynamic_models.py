from typing import Callable

from langchain.agents.middleware import (
    wrap_model_call, 
    ModelRequest, 
    ModelResponse,
    AgentMiddleware,
)

from src.langchain_syntax.llm.factory import get_mistral
from src.config import default_model_name, large_model_name


def dynamic_model_factory(
    context_size_threshold: int = 10,
    base_model_name: str = default_model_name,
    large_model_name: str = large_model_name,
) -> AgentMiddleware:
    """Выбирает модель в зависимости от длины диалога"""

    @wrap_model_call
    def model_changer(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:

        new_model_name = (
            large_model_name 
            if len(request.messages) > context_size_threshold 
            else base_model_name
        )

        if request.model.model_name != new_model_name:
            request = request.override(model=get_mistral(new_model_name))

        return handler(request)

    return model_changer
