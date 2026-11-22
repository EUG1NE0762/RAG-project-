# RAG-project-
software foundation

A "Lecture Slides Assistant" built step by step with the same pipeline as `rag-for-beginners`:
LangChain + embeddings + a persistent Chroma vector store, over the PDFs in `slides/`, with Gemini writing the answers.

**Embeddings run locally by default** (`BAAI/bge-small-en-v1.5` via FastEmbed): free, offline, no daily limit.
The model (~130 MB) downloads into `models/` on first use. To use Gemini embeddings instead, put
`EMBEDDING_PROVIDER=gemini` in `.env` and re-run step 1 (each provider has its own database folder).

**Chat models** are tried in order (`CHAT_MODELS` in `gemini_setup.py`). Gemini's free tier limits each model
per day, so when one is used up the next one answers.

## The steps

| Script | What it does |
|---|---|
| `gemini_setup.py` | Shared embeddings (local or Gemini), chat models with fallback, paths, Chroma store, RRF |
| `1_ingestion_pipeline.py` | **Load** every PDF in `slides/` (one document per slide) → **split** with `RecursiveCharacterTextSplitter` → **embed** and store in `db/chroma_db_local` (resumable) |
| `2_retrieval_pipeline.py` | Retrieval methods: similarity search, similarity with score threshold, MMR |
| `3_answer_generation.py` | Retrieve top-k chunks and have Gemini answer using only those chunks |
| `4_history_aware_generation.py` | Terminal chat: rewrites follow-up questions into standalone ones using chat history, then retrieves and answers |
| `5_multi_query_rrf.py` | Multi-query retrieval (LLM writes 3 rephrasings) fused with Reciprocal Rank Fusion |
| `6_hybrid_search.py` | Hybrid search: vector + BM25 keyword retrieval fused with weighted Reciprocal Rank Fusion (same as LangChain's `EnsembleRetriever`) |
| `app.py` | Streamlit chat (history-aware) with a sidebar to pick the retrieval method and k, and the source slides shown under each answer |

## How to run

1. Python 3.10+ and a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Install:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Put your key in `.env`: `GEMINI_API_KEY=your-key-here`
4. Build the vector store (takes about 2 minutes; delete `db/` and re-run after changing slides):
   ```powershell
   python 1_ingestion_pipeline.py
   ```
5. Try the steps (`python 2_retrieval_pipeline.py`, `python 3_answer_generation.py`, ...) or start the chat app:
   ```powershell
   streamlit run app.py
   ```

## Documentation

- `docs/RAG_Project_Documentation.pdf` — full technical documentation: architecture, each step, algorithms with formulas, fixes, performance measurements, how to run, troubleshooting and a glossary of terms.
- `docs/RAG_Project_Presentation.pptx` — 15-slide presentation of the project.
