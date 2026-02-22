import os
import sys
sys.path.append(os.getcwd())
import json
import re

from datasets import load_dataset
from sentence_transformers import CrossEncoder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.langchain_syntax.rag.advanced import rerank_documents
from src.langchain_syntax.llm.factory import get_mistral
from src.config import reranker_model_name


class RAG:

    def __init__(
        self,
        encoder_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        hugging_face_dataset: str = "BI55/MedText",
        db_dir: str = "./src/data/rag",
        use_reranker: bool = True,
        **retriever_kwargs,
    ):
        self.llm = get_mistral()

        self.init_db(
            encoder_model_name=encoder_model_name,
            hugging_face_dataset=hugging_face_dataset,
            db_dir=db_dir,
        )
        self.init_retriever_chain(use_reranker=use_reranker, **retriever_kwargs)

        self.prompt = ChatPromptTemplate.from_template(base_prompt)

        self.rag = (
            {
                "context": self.retriever_chain,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )


    def init_db(self, encoder_model_name: str, hugging_face_dataset: str, db_dir: str):
        embeddings = HuggingFaceEmbeddings(
            model_name=encoder_model_name, 
            encode_kwargs={"normalize_embeddings": True},
        )

        collection_name = re.sub(r"[^a-zA-Z0-9._-]", "", hugging_face_dataset)
        if os.path.exists(db_dir):
            self.db = Chroma(
                persist_directory=db_dir, 
                collection_name=collection_name,
                embedding_function=embeddings,
            )
        else:
            d = load_dataset(hugging_face_dataset)

            docs = [
                Document(
                    page_content=v["Prompt"], 
                    metadata={"R": v["Completion"]},
                ) for v in d["train"]
            ]

            self.db = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=db_dir,
            )
    

    def init_retriever_chain(self, use_reranker: bool, **retriever_kwargs):
        self.retriever = self.db.as_retriever(**retriever_kwargs)

        if use_reranker:
            reranker = CrossEncoder(reranker_model_name)

            def retrieve(query):
                docs = self.retriever.invoke(query)

                return "\n\n".join(
                    rerank_documents(
                        reranker_model=reranker,
                        query=query,
                        docs=[format_single_doc(d) for d in docs],
                        top_n=5,
                    )
                )

            self.retriever_chain = RunnableLambda(retrieve)
        
        else:
            self.retriever_chain = self.retriever | format_retriever_output
    
    
    def invoke(self, query: str) -> str:
        return self.rag.invoke(query)



def format_single_doc(doc: Document):
    return f"Patient: {doc.page_content}\nConclusion:{doc.metadata['R']}"


def format_retriever_output(docs: list[Document]):
    return "\n\n".join(format_single_doc(d) for d in docs)


base_prompt = """
You're a useful medical assistant.
Answer the question using information only from the contexts of similar situations.

Context:
{context}

Question:
{question}
"""




if __name__ == "__main__":

    rag = RAG(
        use_reranker=True,
        # Retriever kwargs:
        search_type="mmr", 
        search_kwargs={"k": 10, "lambda_mult": 0.25, "fetch_k": 50},
    )

    user_query = "Headache every day, can't sleep, constant dry mouth. I don't use drugs, alcohol, or cigarettes."
    rag_answer = rag.invoke(user_query)
    contexts = rag.retriever_chain.invoke(user_query)

    with open(os.path.join(os.path.dirname(__file__), "base_rag.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "user_query": user_query, 
                "rag_answer": rag_answer,
                "contexts": contexts,
            }, 
            f, 
            indent=4, 
            ensure_ascii=False,
        )
