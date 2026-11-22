"""
app.py — browser chat for the lecture-slides RAG pipeline (Streamlit).

Combines every step: history-aware question rewriting (step 4), a choice of retrieval method
(steps 2, 5, 6), grounded answer generation (step 3), streamed answers and source citations.

Run with:  venv\\Scripts\\streamlit run app.py   (after venv\\Scripts\\python.exe 1_ingestion_pipeline.py)

How Streamlit works: the whole script re-runs top to bottom on every interaction (each
question, each slider move). st.session_state keeps data between runs (the chat), and
@st.cache_resource keeps expensive objects (models, database) loaded once.
"""
import os

import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.retrievers import BM25Retriever
from gemini_setup import get_chat_model, get_vector_store, describe, reciprocal_rank_fusion, quota_message, generate_query_variations, PERSIST_DIR

st.set_page_config(page_title="Lecture Slides Assistant", page_icon="📚")
st.title("📚 Lecture Slides Assistant")

# Retrieval methods offered in the sidebar
METHODS = ["Similarity", "MMR (diverse)", "Multi-query + RRF", "Hybrid (vector + BM25)"]


@st.cache_resource
def load_resources():
    """Load the vector store, chat model and BM25 keyword index ONCE per server, not per question."""
    db = get_vector_store()
    # Read all stored chunks back out of Chroma (BM25 needs the raw texts)
    stored = db.get(include=["documents", "metadatas"])
    all_docs = [Document(page_content=t, metadata=m or {})
                for t, m in zip(stored["documents"], stored["metadatas"])]
    bm25 = BM25Retriever.from_documents(all_docs)   # keyword index built once, not per question
    return db, get_chat_model(), all_docs, bm25


# Stop early with a clear message if step 1 hasn't been run
if not os.path.exists(PERSIST_DIR):
    st.error("No vector store found. Run `python 1_ingestion_pipeline.py` first.")
    st.stop()

try:
    db, model, all_docs, bm25 = load_resources()
except RuntimeError as e:          # e.g. missing API key
    st.error(str(e))
    st.stop()

if not all_docs:
    st.error("The vector store is empty. Run `python 1_ingestion_pipeline.py` first.")
    st.stop()


# ---------- Retrieval methods (steps 2, 9, 10, 11, 12 of rag-for-beginners) ----------

def retrieve(query, method, k):
    """Return the k most relevant chunks for `query` using the chosen retrieval method."""
    if method == "MMR (diverse)":
        # Relevant AND different from each other: pick k out of a larger candidate pool
        return db.as_retriever(search_type="mmr",
                               search_kwargs={"k": k, "fetch_k": max(10, 3 * k), "lambda_mult": 0.5}).invoke(query)
    if method == "Multi-query + RRF":
        # Search with the question plus 3 LLM rephrasings, then fuse the ranked lists
        variations = generate_query_variations(model, query)
        retriever = db.as_retriever(search_kwargs={"k": 5})
        results = [retriever.invoke(q) for q in [query] + variations]
        return [d for d, _ in reciprocal_rank_fusion(results)][:k]
    if method == "Hybrid (vector + BM25)":
        # Meaning-based + exact-keyword search, fused with weights 0.7 / 0.3
        bm25.k = k
        vector = db.as_retriever(search_kwargs={"k": k}).invoke(query)
        fused = reciprocal_rank_fusion([vector, bm25.invoke(query)], weights=[0.7, 0.3])
        return [d for d, _ in fused][:k]
    # Default: plain similarity search (top k by cosine similarity)
    return db.as_retriever(search_kwargs={"k": k}).invoke(query)


# ---------- Sidebar ----------

with st.sidebar:
    st.header("Settings")
    method = st.selectbox("Retrieval method", METHODS)
    k = st.slider("Chunks to retrieve (k)", 1, 10, 4)   # how many chunks go into the prompt
    st.caption(f"Chunks in database: {len(all_docs)}")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("**Try asking:**")
    st.markdown(
        "- What is defect density?\n"
        "- What is a software configuration item?\n"
        "- How is quality assurance different from quality control?"
    )


def show_sources(sources):
    """Collapsible list of the slides an answer was based on."""
    with st.expander(f"Sources ({len(sources)})"):
        for i, (label, text) in enumerate(sources, 1):
            st.markdown(f"**{i}. {label}**")
            st.text(text[:800])


# The chat so far, kept across reruns: list of dicts {role, content, sources, search}
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-draw the previous messages (Streamlit redraws the whole page on every run)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("search"):
            st.caption(f"Searched for: {m['search']}")
        if m.get("sources"):
            show_sources(m["sources"])


MAX_HISTORY = 6   # last 3 question/answer pairs: enough context, keeps prompts small and fast


def history_as_messages():
    """Recent chat turns as LangChain messages (HumanMessage = user, AIMessage = assistant)."""
    return [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages[-MAX_HISTORY:]]


def text_chunks(stream):
    """Yield just the text from model output chunks.

    Some Gemini models return content as a plain string, others as a list of parts
    ([{'type': 'text', 'text': ...}]); this handles both.
    """
    for chunk in stream:
        c = chunk.content
        yield c if isinstance(c, str) else "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)


question = st.chat_input("Ask about your lecture units")
if question:
    history = history_as_messages()                 # history BEFORE this question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching slides..."):
                # Step 1: make the question standalone using chat history (only needed for follow-ups)
                search_question = question
                if history:
                    rewrite = model.invoke(
                        [SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question.")]
                        + history
                        + [HumanMessage(content=f"New question: {question}")]
                    )
                    search_question = next(text_chunks([rewrite])).strip() or question

                # Step 2: retrieve relevant chunks with the chosen method
                docs = retrieve(search_question, method, k)
                context = "\n".join(f"- [{describe(d)}] {d.page_content}" for d in docs)

            # Step 3: answer from the documents, STREAMED so words appear as they are generated
            # (the user sees the first words in ~1s instead of waiting for the whole answer)
            prompt = f"""Based on the following documents, please answer this question: {question}

Documents:
{context}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
            answer = st.write_stream(text_chunks(model.stream(
                [SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history.")]
                + history
                + [HumanMessage(content=prompt)]
            )))
        except Exception as e:
            # Quota errors get an accurate explanation; anything else is shown briefly
            answer = quota_message(e) or f"⚠️ Error: {str(e)[:500]}"
            docs, search_question = [], None
            st.markdown(answer)

        # Show what was actually searched (if rewritten) and which slides were used
        if search_question and search_question != question:
            st.caption(f"Searched for: {search_question}")
        sources = [(describe(d), d.page_content) for d in docs]
        if sources:
            show_sources(sources)

    # Save the assistant turn so it is redrawn on the next run and used as history
    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources,
        "search": search_question if search_question != question else None,
    })
