"""
STEP 2 — Retrieval: find the slide chunks most relevant to a question (no LLM involved yet).

The question is embedded with the same model used in step 1, and Chroma returns the stored
chunks whose vectors are closest to it. Three retrieval methods are compared below.

Run:  venv\\Scripts\\python.exe 2_retrieval_pipeline.py
"""
from dotenv import load_dotenv
from gemini_setup import get_vector_store, describe

load_dotenv()

# Open the vector store built by 1_ingestion_pipeline.py
db = get_vector_store()

# The question to search for
query = "What is defect density and how is it calculated?"
print(f"User Query: {query}\n")

# ──────────────────────────────────────────────────────────────────
# METHOD 1: Basic Similarity Search
# Returns the k chunks with the highest cosine similarity to the question.
# ──────────────────────────────────────────────────────────────────
print("=== METHOD 1: Similarity Search (k=5) ===")
retriever = db.as_retriever(search_kwargs={"k": 5})   # a Retriever: question in → list of Documents out
for i, doc in enumerate(retriever.invoke(query), 1):
    print(f"Document {i} [{describe(doc)}]:\n{doc.page_content}\n")
print("-" * 60)

# ──────────────────────────────────────────────────────────────────
# METHOD 2: Similarity with Score Threshold
# Same search, but drops chunks whose similarity score is below the threshold
# (0 = unrelated, 1 = identical). Useful to return nothing for off-topic questions.
# ──────────────────────────────────────────────────────────────────
print("\n=== METHOD 2: Similarity with Score Threshold (0.3) ===")
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3}
)
docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents:")
for i, doc in enumerate(docs, 1):
    print(f"  Document {i} [{describe(doc)}]: {doc.page_content[:120]}...")
print("-" * 60)

# ──────────────────────────────────────────────────────────────────
# METHOD 3: Maximum Marginal Relevance (MMR)
# First fetches fetch_k candidates, then picks k of them that are relevant to the question
# but different from each other — avoids returning several near-duplicate slides.
# lambda_mult: 1 = pure relevance, 0 = pure diversity.
# ──────────────────────────────────────────────────────────────────
print("\n=== METHOD 3: Maximum Marginal Relevance (MMR, λ=0.5) ===")
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)
for i, doc in enumerate(retriever.invoke(query), 1):
    print(f"  Document {i} [{describe(doc)}]: {doc.page_content[:120]}...")


# Synthetic test questions (replace `query` above with any of these):
# 1. "What is a software configuration item?"
# 2. "What is the difference between quality assurance and quality control?"
# 3. "Why learn software metrics?"
# 4. "What happens during the prototype phase of product maturity?"
# 5. "How does configuration management protect quality?"
# 3. "Why learn software metrics?"
