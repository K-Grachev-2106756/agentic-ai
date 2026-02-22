# SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE THROUGH SELF-REFLECTION
# https://arxiv.org/pdf/2310.11511
import os
import sys
sys.path.append(os.getcwd())
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from src.langchain_syntax.llm.factory import get_mistral
from src.langchain_syntax.rag.base_rag import RAG, format_single_doc


class SelfRAG(RAG):

    def __init__(
        self,
        encoder_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        hugging_face_dataset: str = "BI55/MedText",
        db_dir: str = "./src/data/rag",
    ):
        self.llm = get_mistral()
        self.history = []

        self.init_critique_chain()
        self.init_document_relevance_chain()
        self.init_single_generation_chain()
        self.init_generation_chain()
        self.init_retrieve_necessity_chain()
        self.init_no_context_generation_chain()

        self.init_db(
            encoder_model_name=encoder_model_name,
            hugging_face_dataset=hugging_face_dataset,
            db_dir=db_dir,
        )


    def init_retrieve_necessity_chain(self):
        retrieve_necessity_prompt_templ = ChatPromptTemplate.from_template(retrieve_necessity_prompt)
        self.retrieve_chain = retrieve_necessity_prompt_templ | self.llm | StrOutputParser()


    def init_document_relevance_chain(self):
        document_relevance_prompt_templ = ChatPromptTemplate.from_template(document_relevance_prompt)
        self.relevance_chain = document_relevance_prompt_templ | self.llm | StrOutputParser()


    def init_single_generation_chain(self):
        single_generation_prompt_templ = ChatPromptTemplate.from_template(single_generation_prompt)
        self.single_generation_chain = single_generation_prompt_templ | self.llm | StrOutputParser()


    def init_generation_chain(self):
        generation_prompt_templ = ChatPromptTemplate.from_template(generation_prompt)
        self.generation_chain = generation_prompt_templ | self.llm | StrOutputParser()
    

    def init_critique_chain(self):
        critique_prompt_templ = ChatPromptTemplate.from_template(critique_prompt)
        self.critique_chain = critique_prompt_templ | self.llm | JsonOutputParser()
    
    
    def init_no_context_generation_chain(self):
        no_context_generation_prompt_templ = ChatPromptTemplate.from_template(no_context_generation_prompt)
        self.no_context_generation_chain = no_context_generation_prompt_templ | self.llm | StrOutputParser()


    def invoke(self, question: str, max_retries: int = 2, top_k: int = 3) -> str:
        candidates, best_contexts = {"fully": [], "partially": []}, []

        for i_retry in range(max_retries):
            print(f"Try #{i_retry + 1}")

            # 1. Decide retrieve
            retrieve_decision = self.retrieve_chain.invoke(
                {
                    "question": question, 
                    "context": "\n\n".join(best_contexts) or "None",
                }
            ).strip()

            if retrieve_decision == "YES":

                # 2. Retrieve documents
                docs = (
                    self.db.as_retriever(
                        search_type="similarity", 
                        search_kwargs={"k": top_k * (i_retry + 1)},
                    )
                    .invoke(question)[top_k * i_retry:]
                )

                print(f"recieved {len(docs)} docs")

                for d in docs:
                    context = format_single_doc(d)

                    # 3. ISREL
                    isrel = self.relevance_chain.invoke({"question": question, "doc": context}).strip()

                    if isrel != "YES":
                        continue
                    
                    for _ in range(max_retries):
                        # 4. Generate answer conditioned on doc
                        answer = self.single_generation_chain.invoke({"question": question, "doc": context})

                        # 5. Critique
                        critique = self.critique_chain.invoke({"question": question, "answer": answer, "doc": context})

                        if critique["SUPPORT"] != "no":
                            break
                    
                    if critique["SUPPORT"] == "no":
                        continue
                    
                    candidates[critique["SUPPORT"]].append(
                        (int(critique["USEFULNESS"]), answer)
                    )
                
                self.history = candidates
                print(f"total candidates: {sum(len(v) for v in candidates.values())}")
                print(f"candidates:\n{candidates}")

                best_contexts = []  # best_contexts will be updated every i_retry iteration
                for support in ["fully", "partially"]:
                    if candidates[support]:
                        candidates[support].sort(key=lambda x: x[0], reverse=True)
                        
                        usefulness = candidates[support][0][0]
                        for i_usefulness, i_answer in candidates[support]:
                            if i_usefulness != usefulness:  # best contexts is in best_contexts already
                                retrieve_decision = self.retrieve_chain.invoke(
                                    {
                                        "question": question, 
                                        "context": "\n\n".join(best_contexts) or "None",
                                    }
                                ).strip()

                                if retrieve_decision == "NO":
                                    return self.generation_chain.invoke({
                                        "question": question,
                                        "context": "\n\n".join(best_contexts),
                                    })
                                
                                else:
                                    usefulness -= 1  # answers with lower quality will be added to the best_contexts
                            
                            best_contexts.append(i_answer)
                
                print(f"Not enough info with contexts: {'\n'.join(best_contexts)}.\n\nRetrying...")

            elif len(best_contexts):
                print("Generation with retrieved contexts...")
                return self.generation_chain.invoke({
                    "question": question,
                    "context": "\n\n".join(best_contexts),
                })
            
            else:
                # 6. Fallback (no retrieve)
                print("No retrived contexts answer will be given")
                return self.no_context_generation_chain.invoke({"question": question})

        print("The attempts are over. Generation with retrieved contexts...")
        return self.generation_chain.invoke({
            "question": question,
            "context": "\n\n".join(best_contexts) or "None",
        })


retrieve_necessity_prompt = """
Given the input:
Question:
{question}

Found contexts:                                                       
{context}

Knowledge base provides information about patients and assumptions about their illnesses.
Should we retrieve external documents to answer this question?
If the question has nothing to do with health, answer "NO".
Answer only: YES or NO."""


document_relevance_prompt = """
Question:
{question}

Perhaps a similar case:
{doc}

Is this case relevant for answering the patient's question?
Answer only YES or NO."""


single_generation_prompt = """
Question:
{question}

Similar case:
{doc}

Don't continue the dialogue with unnecessary questions.
Generate an answer grounded in the similar case."""


generation_prompt = """
Patient's question:
{question}

Similar cases:
{context}

Generate an answer grounded in the similar cases.
"""


critique_prompt = """
Question:
{question}

Answer:
{answer}

Document:
{doc}

Is the answer supported by the document? (fully/partially/no)
Evaluate the usefulness of the document to answer the question from 1 to 5, where 5 is the most useful.

Return correct JSON-serializable dict. For example:
{{
    "SUPPORT": Literal["fully", "partially", "no"], 
    "USEFULNESS": Literal[1, 2, 3, 4, 5]
}}"""


no_context_generation_prompt = """
Patient's question:
{question}

Answer the question directly.
If the question has nothing to do with health, answer "I can't answer your question."
"""




if __name__ == "__main__":
    rag = SelfRAG()

    user_query = "I'm 24 years old young woman. I do not smoking and drinking, I don't take medications. Excessive hair loss, lethargy, bags under the eyes, gray skin tone. What are the possible causes of my ailments?"
    rag_answer = rag.invoke(user_query, max_retries=3)

    with open(os.path.join(os.path.dirname(__file__), "self_rag.json"), "w", encoding="utf-8") as f:
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
