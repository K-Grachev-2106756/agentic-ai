from langchain_mistralai import ChatMistralAI
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import mistral_api_key


def get_mistral(model_name: str) -> BaseChatModel:
    return ChatMistralAI(
        name=model_name,
        api_key=mistral_api_key,
    )
