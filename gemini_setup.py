"""
gemini_setup.py — shared configuration and helpers for every step of the RAG pipeline.

RAG (Retrieval-Augmented Generation) = first RETRIEVE the slide chunks most relevant
to a question, then let a large language model GENERATE an answer using only those chunks.

This module is imported by the numbered scripts (1_ ... 6_) and by app.py so that they
all use the same models, the same database folder and the same helper functions.

Your Gemini API key goes in a .env file in this folder as either
    GEMINI_API_KEY=...
or  GOOGLE_API_KEY=...
"""
import os
from dotenv import load_dotenv                      # reads KEY=value lines from .env into os.environ
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()   # make the .env values (API key, EMBEDDING_PROVIDER) visible via os.getenv

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────

# Which embedding model turns text into vectors (lists of numbers that capture meaning):
#   "local"  → BAAI/bge-small-en-v1.5 running on this computer via FastEmbed:
#              free, offline, no daily limit (the default).
#   "gemini" → Google's gemini-embedding-001 API (free tier = 1,000 requests/day).
# Override with EMBEDDING_PROVIDER=gemini in .env. Each provider gets its own database
# folder, because vectors produced by different models are not comparable.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"      # 384-number vectors, ~130 MB download
EMBEDDING_MODEL = "models/gemini-embedding-001"        # used only when EMBEDDING_PROVIDER=gemini

# Chat (answer-writing) models, tried in order by get_chat_model(). The Gemini free tier
# limits each model per day, so if one is used up, busy or too slow, the next one answers.
# Ordered fastest first (measured: flash-lite ~1.4s, 2.5-flash ~1.7s, 3.5-flash ~24s).
CHAT_MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-3.5-flash"]

# Shared paths so every step reads/writes the same place
SLIDES_PATH = "slides"                                  # input: lecture PDFs
PERSIST_DIR = "db/chroma_db" if EMBEDDING_PROVIDER == "gemini" else "db/chroma_db_local"  # output: vector DB


# ──────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────

def _api_key():
    """Return the Gemini API key from the environment / .env, or explain how to add one."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Add GEMINI_API_KEY=... to a .env file in this folder."
        )
    return key


class _CachedQueryEmbeddings(GoogleGenerativeAIEmbeddings):
    """Gemini embeddings that remember each question's vector.

    Asking the same question again reuses the stored vector instead of spending
    another request from the daily quota. (Only used with EMBEDDING_PROVIDER=gemini.)
    """

    def embed_query(self, text, *args, **kwargs):
        cache = self.__dict__.setdefault("_query_cache", {})   # {question text: vector}
        if text not in cache:
            cache[text] = super().embed_query(text, *args, **kwargs)
        return cache[text]


def get_embeddings():
    """Return the embedding model selected by EMBEDDING_PROVIDER."""
    if EMBEDDING_PROVIDER == "gemini":
        return _CachedQueryEmbeddings(model=EMBEDDING_MODEL, google_api_key=_api_key())
    # Local model: downloaded into ./models the first time, then runs fully offline
    from langchain_community.embeddings import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model_name=LOCAL_EMBEDDING_MODEL, cache_dir="models")


def get_chat_model(**kwargs):
    """Return a chat model that falls back through CHAT_MODELS.

    `with_fallbacks` tries the first model; if it raises an error (quota used up = HTTP 429,
    overloaded = 503, or the timeout below), LangChain automatically retries with the next.
    Extra keyword arguments (e.g. temperature=0) are passed to every model.
    """
    kwargs.setdefault("max_retries", 0)   # don't retry the same model: move to the next one at once
    kwargs.setdefault("timeout", 30)      # a stuck model hands over after 30s instead of hanging
    models = [ChatGoogleGenerativeAI(model=m, google_api_key=_api_key(), **kwargs) for m in CHAT_MODELS]
    return models[0].with_fallbacks(models[1:]) if len(models) > 1 else models[0]


def get_vector_store():
    """Open the persisted Chroma vector store created by 1_ingestion_pipeline.py.

    "hnsw:space": "cosine" tells Chroma to compare vectors by cosine similarity
    (the angle between them), the usual measure for text embeddings.
    """
    from langchain_chroma import Chroma
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),   # used to embed each question at search time
        collection_metadata={"hnsw:space": "cosine"},
    )


# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────

def describe(doc):
    """Human-readable source label for a chunk, e.g. 'UNIT 1.pdf, slide 3'.

    PyPDFLoader stores the file path in metadata['source'] and a 0-based page number
    in metadata['page'], so 1 is added to match the slide numbers students see.
    """
    name = os.path.basename(doc.metadata.get("source", "?"))
    page = doc.metadata.get("page")
    return f"{name}, slide {page + 1}" if isinstance(page, int) else name


def quota_message(error):
    """Turn a Gemini 'quota exceeded' (HTTP 429) error into an accurate message for the user.

    Per-minute limits clear in under a minute; per-day limits ('PerDay' in the error) don't,
    so the message says which one was hit and how long Google says to wait.
    Returns None for any other kind of error.
    """
    import re
    msg = str(error)
    if not ("429" in msg or "RESOURCE_EXHAUSTED" in msg):
        return None
    wait = re.search(r"retry in ((?:\d+h)?(?:\d+m)?[\d.]+s)", msg)          # e.g. "retry in 43m51.07s"
    wait = f" Google says to retry in {wait.group(1).split('.')[0]}s." if wait else ""
    which = "embedding (search)" if "mbed" in msg else "chat"
    if "PerDay" in msg:
        return (f"⚠️ The daily Gemini free-tier {which} quota is used up, so waiting a minute won't help."
                f"{wait} It resets daily (midnight US Pacific), or use a paid API key.")
    return f"⚠️ Gemini per-minute {which} rate limit hit (free tier).{wait or ' Wait a minute and try again.'}"


def reciprocal_rank_fusion(chunk_lists, weights=None, k=60):
    """Reciprocal Rank Fusion (RRF): merge several ranked result lists into one ranking.

    Every chunk earns  weight / (k + rank)  from each list it appears in (rank 1 = best),
    and the scores are added up. So a chunk ranked highly in several lists rises to the top.
    k = 60 is the standard constant: it softens the gap between rank 1 and rank 2.

    Used for:
      • multi-query retrieval (5_multi_query_rrf.py) — equal weights for every query variation
      • hybrid search (6_hybrid_search.py) — vector 0.7 + BM25 0.3, the same formula
        LangChain's EnsembleRetriever uses.
    Returns a list of (doc, score) pairs, best first.
    """
    from collections import defaultdict
    weights = weights or [1.0] * len(chunk_lists)
    scores, unique = defaultdict(float), {}            # chunk text → total score / chunk object
    for chunks, w in zip(chunk_lists, weights):
        for position, chunk in enumerate(chunks, 1):   # position starts at 1 (the best result)
            unique[chunk.page_content] = chunk         # same text from different lists = same chunk
            scores[chunk.page_content] += w / (k + position)
    return sorted(((unique[c], s) for c, s in scores.items()), key=lambda x: x[1], reverse=True)


def generate_query_variations(llm, original_query, n=3):
    """Multi-query step: ask the LLM for n differently worded versions of the question.

    Different wordings match different slides, which improves recall (finding everything
    relevant). Plain text lines are requested (rather than structured JSON output) because
    Gemini sometimes returned only the original question in structured mode.
    """
    import re
    prompt = f"""Generate {n} different variations of this query that would help retrieve relevant documents:

Original query: {original_query}

Write exactly {n} alternative queries, one per line, that rephrase or approach the same question from different angles.
Word each one differently from the original. No numbering, no extra text."""
    # Strip any "1." / "2)" / "-" prefixes the model adds anyway
    lines = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", l).strip() for l in llm.invoke(prompt).content.splitlines()]
    # Drop blank lines and exact copies of the original question
    return [l for l in lines if l and l.lower() != original_query.strip().lower()][:n]
