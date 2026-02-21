from typing import Union

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


def rerank_documents(
        reranker_model: CrossEncoder, 
        query: str, 
        docs: Union[list[Document], list[str]], 
        top_n: int = 3,
    ):
    texts = [doc.page_content for doc in docs] if isinstance(docs[0], Document) else docs
    
    pairs = [(query, doc) for doc in texts]
    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_n]]
