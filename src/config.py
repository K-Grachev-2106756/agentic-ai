import os

from dotenv import load_dotenv


load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

default_model_name = os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest")
large_model_name = "mistral-large-latest"
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
