# RAG-Fusion: a New Take on Retrieval-Augmented Generation
# https://arxiv.org/html/2402.03367v2
import os
import sys
sys.path.append(os.getcwd())
import json
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from src.langchain_syntax.llm.factory import get_mistral
from src.langchain_syntax.rag.advanced import generate_additional_queries
from src.langchain_syntax.rag.base_rag import RAG, format_single_doc


class FusionRAG(RAG):

    def __init__(
        self,
        encoder_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        hugging_face_dataset: str = "BI55/MedText",
        db_dir: str = "./src/data/rag",
    ):
        self.llm = get_mistral()
        self.history = []

        self.init_db(
            encoder_model_name=encoder_model_name,
            hugging_face_dataset=hugging_face_dataset,
            db_dir=db_dir,
        )

        self.init_retriever_chain(
            use_reranker=False, 
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        self.init_generation_chain()


    def init_generation_chain(self):
        generation_prompt_templ = ChatPromptTemplate.from_template(generation_prompt)
        self.generation_chain = generation_prompt_templ | self.llm | StrOutputParser()
    
    
    @staticmethod
    def rrfscore(rank: int, smooth_rank: float) -> float:
        return 1 / (rank + smooth_rank)


    def invoke(self, question: str, top_k: int = 4, smooth_rank: int = 20) -> str:  # for medium smoothing of the relevance score
        extra_questions = generate_additional_queries(
            model=self.llm,
            prompt=question,
            num_queries=3,
        )

        print(f"Extra questions:\n{'\n'.join(extra_questions)}")

        scores = defaultdict(float)
        for q in set([question] + extra_questions):
            for rank, d in enumerate(self.retriever.invoke(q), 1):
                scores[format_single_doc(d)] += FusionRAG.rrfscore(rank, smooth_rank)
        
        print(f"Unique documents found: {len(scores)}")

        scores = list(scores.items())
        scores.sort(key=lambda x: x[1], reverse=True)
        best_contexts = "\n\n".join([context for context, _ in scores[:top_k]])
        self.history = scores
        
        print(f"Best contexts:\n{best_contexts}")
        
        return self.generation_chain.invoke({"question": question, "context": best_contexts})


generation_prompt = """
Patient's question:
{question}

Similar cases:
{context}

Generate an answer grounded in the similar cases.
"""




if __name__ == "__main__":
    rag = FusionRAG()

    user_query = "I'm 24 years old young woman. I do not smoking and drinking, I don't take medications. Excessive hair loss, lethargy, bags under the eyes, gray skin tone. What are the possible causes of my ailments?"
    rag_answer = rag.invoke(user_query)

    with open(os.path.join(os.path.dirname(__file__), "fusion_rag.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "user_query": user_query, 
                "rag_answer": str(rag_answer),
                "contexts": rag.history,
            }, 
            f, 
            indent=4, 
            ensure_ascii=False,
        )
