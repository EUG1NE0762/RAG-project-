"""
STEP 6 — Hybrid search: vector (semantic) search + BM25 (keyword) search (rag-for-beginners step 12).

  • Vector search matches MEANING ("change control" ≈ "controlled change") but can miss exact
    terms, acronyms and codes.
  • BM25 matches exact WORDS (great for "SCI", "DRE", "KLOC") but doesn't understand synonyms.
Hybrid search runs both and merges them with weighted Reciprocal Rank Fusion
(vector weight 0.7, BM25 weight 0.3) — the same formula as LangChain's EnsembleRetriever.

Run:  venv\\Scripts\\python.exe 6_hybrid_search.py
"""
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.retrievers import BM25Retriever
from gemini_setup import get_vector_store, get_chat_model, describe, reciprocal_rank_fusion

load_dotenv()

db = get_vector_store()


def hybrid_search(query, k=5, weights=(0.7, 0.3)):
    """Return chunks ranked by weighted RRF of vector search and BM25 keyword search."""
    # BM25 needs the raw chunk texts, so read them back out of Chroma (same chunks as step 1)
    stored = db.get(include=["documents", "metadatas"])
    documents = [Document(page_content=t, metadata=m or {})
                 for t, m in zip(stored["documents"], stored["metadatas"])]

    vector_retriever = db.as_retriever(search_kwargs={"k": k})   # semantic: cosine similarity
    bm25_retriever = BM25Retriever.from_documents(documents)     # keyword: term frequency scoring
    bm25_retriever.k = k

    # Each retriever gives its own ranked list; RRF merges them using the weights
    fused = reciprocal_rank_fusion([vector_retriever.invoke(query), bm25_retriever.invoke(query)], list(weights))
    return [doc for doc, _ in fused]


if __name__ == "__main__":
    # Exact acronym ("SCI") for BM25 + a concept ("change control") for vector search
    test_query = "SCI configuration item change control"
    retrieved_chunks = hybrid_search(test_query)
    print(f"Query: {test_query}\n")
    for i, doc in enumerate(retrieved_chunks, 1):
        print(f"{i}. [{describe(doc)}] {doc.page_content[:150]}...")

    # Generate an answer from the hybrid results (same prompt pattern as step 3)
    combined_input = f"""Based on the following documents, please answer this question: {test_query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in retrieved_chunks])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
    result = get_chat_model().invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ])
    print("\n--- Generated Response ---")
    print(result.content)
