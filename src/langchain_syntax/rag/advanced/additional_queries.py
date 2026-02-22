from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser


def generate_additional_queries(model: BaseChatModel, prompt: str, num_queries: int = 3) -> list[str]:
    query_generation_prompt = ChatPromptTemplate.from_template("""
Given the prompt: '{prompt}', 

generate {num_queries} questions that are better articulated. 

Return correct JSON-serializable list[str]. For example: 
['question 1', 'question 2', 'question 3']
""")
    query_generation_chain = (
        query_generation_prompt 
        | model
        | JsonOutputParser()
    )
    
    return query_generation_chain.invoke({"prompt": prompt, "num_queries": num_queries})
