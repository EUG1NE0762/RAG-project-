"""
STEP 5 — Multi-query retrieval + Reciprocal Rank Fusion (rag-for-beginners steps 10 and 11).

Problem: one wording of a question can miss slides that use different words for the same idea.
Solution:
    1. Multi-query: the LLM writes 3 different wordings of the question
    2. search with each wording (each gives its own ranked list of chunks)
    3. Reciprocal Rank Fusion (RRF): merge the lists — a chunk scores 1/(60 + rank) per list
       it appears in, so chunks found by several wordings rise to the top
This improves recall (finding all relevant slides) at the cost of one extra LLM call.

Run:  venv\\Scripts\\python.exe 5_multi_query_rrf.py
"""
from dotenv import load_dotenv
from gemini_setup import get_vector_store, get_chat_model, describe, reciprocal_rank_fusion, generate_query_variations

load_dotenv()

db = get_vector_store()
llm = get_chat_model(temperature=0)   # temperature 0 = most predictable output


if __name__ == "__main__":
    original_query = "How do we measure software quality?"
    print(f"Original Query: {original_query}\n")

    # Step 1: Generate multiple query variations with the LLM
    query_variations = generate_query_variations(llm, original_query)
    print("Generated Query Variations:")
    for i, v in enumerate(query_variations, 1):
        print(f"{i}. {v}")
    print("\n" + "=" * 60)

    # Step 2: Search with each variation and keep every ranked list for RRF
    retriever = db.as_retriever(search_kwargs={"k": 5})
    all_retrieval_results = []
    for i, q in enumerate(query_variations, 1):
        docs = retriever.invoke(q)
        all_retrieval_results.append(docs)
        print(f"\n=== RESULTS FOR QUERY {i}: {q} ===")
        for j, doc in enumerate(docs, 1):
            print(f"  {j}. [{describe(doc)}] {doc.page_content[:100]}...")

    # Step 3: Fuse the lists with Reciprocal Rank Fusion (k=60)
    fused = reciprocal_rank_fusion(all_retrieval_results, k=60)

    # Step 4: Show the final fused ranking
    print("\n" + "=" * 60)
    print("FINAL RRF RANKING")
    print("=" * 60)
    for rank, (doc, score) in enumerate(fused[:10], 1):
        print(f"RANK {rank} (RRF Score: {score:.4f}) [{describe(doc)}]")
        print(f"{doc.page_content[:200]}...")
        print("-" * 50)
    print(f"\nFused {len(fused)} unique chunks from {len(query_variations)} query variations.")
