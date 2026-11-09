"""Answer generation from retrieved documents (retrieval-augmented generation)."""

from rag_search.config import LLM_DOCUMENT_CHARS, SERVICE_NAME
from rag_search.llm import ask
from rag_search.util import format_documents


def generate_answer(query: str, results: list[dict]) -> str:
    """Answer a query in natural language using the retrieved documents."""
    docs = format_documents(results, LLM_DOCUMENT_CHARS)
    prompt = f"""You are a RAG agent for {SERVICE_NAME}, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""
    return ask(prompt)


def summarize_results(query: str, results: list[dict]) -> str:
    """Synthesize a short summary across the retrieved documents."""
    docs = format_documents(results, LLM_DOCUMENT_CHARS)
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to {SERVICE_NAME} users. {SERVICE_NAME} is a movie streaming service.

Query: {query}

Search results:
{docs}

Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:"""
    return ask(prompt)


def answer_with_citations(query: str, results: list[dict]) -> str:
    """Answer a query and cite the numbered source documents as [1], [2], ..."""
    docs = format_documents(results, LLM_DOCUMENT_CHARS, numbered=True)
    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of {SERVICE_NAME}, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    return ask(prompt)


def answer_question(question: str, results: list[dict]) -> str:
    """Answer a direct question in a short, conversational tone."""
    docs = format_documents(results, LLM_DOCUMENT_CHARS)
    prompt = f"""Answer the user's question based on the provided movies that are available on {SERVICE_NAME}, a streaming service.

Question: {question}

Documents:
{docs}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
    return ask(prompt)
