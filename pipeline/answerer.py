"""
pipeline/answerer.py — Final answer generation with inline citations.

Stuffs the relevance-graded chunks into a strict system prompt and
calls the Groq LLM to produce a cited answer.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # ✅ verified


FINAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a precise document assistant. Answer using ONLY the context below.\n\n"
            "Rules:\n"
            "1. Use only information directly present in the context.\n"
            "2. Ignore context sections that do not help answer the question.\n"
            "3. Do NOT mention documents unrelated to the answer.\n"
            "4. Cite every fact with: [Document Name, page X]\n"
            "5. If the answer is not in the context, respond with exactly:\n"
            "   \"I don't have enough information in the provided documents.\"\n\n"
            "Context:\n{context}"
        ),
    ),
    ("human", "{input}"),
])


def answer_question(
    question: str,
    relevant_chunks: list[Document],
    llm: ChatGroq,
) -> str:
    """
    Stuff *relevant_chunks* into FINAL_PROMPT and invoke the LLM.

    Args:
        question        : rewritten user question
        relevant_chunks : chunks that passed reranking + relevance grading
        llm             : Groq LLM instance
    """
    chain = create_stuff_documents_chain(llm=llm, prompt=FINAL_PROMPT)
    return chain.invoke({"input": question, "context": relevant_chunks})

