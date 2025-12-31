import os

from dotenv import load_dotenv


load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
