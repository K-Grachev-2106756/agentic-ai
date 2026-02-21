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

from src.langchain_syntax.rag.reranker import rerank_documents
from src.langchain_syntax.llm.factory import get_mistral
from src.config import reranker_model_name


def format_single_doc(doc: Document):
    return f"Patient: {doc.page_content}\nConclusion:{doc.metadata['R']}"


def format_retriever_output(docs: list[Document]):
    return "\n\n".join(format_single_doc(d) for d in docs)


def init_medical_rag(
        encoder_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        hugging_face_dataset: str = "BI55/MedText",
        db_dir: str = "./src/data/rag",
        use_reranker: bool = True,
        return_retriever_chain: bool = False,
    ):

    # RETRIEVER
    embeddings = HuggingFaceEmbeddings(
        model_name=encoder_model_name, 
        encode_kwargs={"normalize_embeddings": True},
    )

    collection_name = re.sub(r"[^a-zA-Z0-9._-]", "", hugging_face_dataset)
    if os.path.exists(db_dir):
        db = Chroma(
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

        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=db_dir,
        )
    

    # RERANKER
    if use_reranker:
        retriever = db.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 10, "lambda_mult": 0.25, "fetch_k": 50},
        )

        reranker = CrossEncoder(reranker_model_name)

        def retrieve(query):
            docs = retriever.invoke(query)

            return "\n\n".join(
                rerank_documents(
                    reranker_model=reranker,
                    query=query,
                    docs=[format_single_doc(d) for d in docs],
                    top_n=5,
                )
            )

        retriever_chain = RunnableLambda(retrieve)
    
    else:
        retriever = db.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 5, "lambda_mult": 0.25, "fetch_k": 25},
        )

        retriever_chain = retriever | format_retriever_output
    

    # LLM
    llm = get_mistral()
    
    # BASE PROMPT
    prompt = ChatPromptTemplate.from_template("""
    You're a useful medical assistant.
    Answer the question using information only from the contexts of similar situations.

    Context:
    {context}

    Question:
    {question}
    """)


    # RAG
    rag = (
        {
            "context": retriever_chain,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    if return_retriever_chain:
        return rag, retriever_chain
    
    return rag


if __name__ == "__main__":

    rag, retriever_chain = init_medical_rag(return_retriever_chain=True)

    user_query = "Headache every day, can't sleep, constant dry mouth. I don't use drugs, alcohol, or cigarettes."
    rag_answer = rag.invoke(user_query)
    contexts = retriever_chain.invoke(user_query)

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
