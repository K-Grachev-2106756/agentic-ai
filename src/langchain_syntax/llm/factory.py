from langchain_openai import ChatOpenAI

from src.config import mistral_api_key, default_model_name


def get_mistral(model_name: str = default_model_name) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        api_key=mistral_api_key,
        base_url="https://api.mistral.ai/v1",
        temperature=0.1,
        top_p=0.9,
    )
