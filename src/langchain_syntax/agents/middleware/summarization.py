from typing import Literal

from langchain.agents.middleware import SummarizationMiddleware

from src.langchain_syntax.llm.factory import get_mistral
from src.config import default_model_name


def summarize_factory(
        model: str = default_model_name,
        trigger: Literal["tokens", "messages"] = "tokens",
        n_trigger: int = 256,
        keep: Literal["tokens", "messages"] = "messages",
        n_keep: int = 1,
    ) -> SummarizationMiddleware:
    """Call dialog summarization after trigger to save the size of the context."""

    return SummarizationMiddleware(
        model=get_mistral(model),
        trigger=(trigger, n_trigger),
        keep=(keep, n_keep),
    )
