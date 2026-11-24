"""
app.py — browser chat for the lecture-slides RAG pipeline (Streamlit), styled with the Cyber UI.

Combines every step: history-aware question rewriting (step 4), a choice of retrieval method
(steps 2, 5, 6), grounded answer generation (step 3), streamed answers and source citations.

Run with:  venv\\Scripts\\streamlit run app.py   (after venv\\Scripts\\python.exe 1_ingestion_pipeline.py)

How Streamlit works: the whole script re-runs top to bottom on every interaction (each
question, each click). st.session_state keeps data between runs (the chat), and
@st.cache_resource keeps expensive objects (models, database) loaded once.

Look and feel: assets/cyber.css (futuristic HUD, the default) + .streamlit/config.toml (theme colours).
The older frosted-glass look is kept in assets/frost.css; set CSS_PATH to it to switch back.
"""
import html
import os

import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.retrievers import BM25Retriever
from gemini_setup import (get_chat_model, get_vector_store, describe, reciprocal_rank_fusion, quota_message,
                          generate_query_variations, PERSIST_DIR, EMBEDDING_PROVIDER, LOCAL_EMBEDDING_MODEL, CHAT_MODELS)

st.set_page_config(page_title="Lecture Slides Assistant", page_icon=":material/menu_book:", layout="centered",
                   initial_sidebar_state="auto")   # auto = collapsed on phones


# ---------- Cyber UI styling ----------

@st.cache_data
def load_css(path, modified):
    """Read the stylesheet once per version; it is injected on every rerun.
    `modified` (the file's timestamp) is part of the cache key, so edits show up on refresh."""
    with open(path, encoding="utf-8") as f:
        return f.read()


CSS_PATH = "assets/cyber.css"
st.markdown(f"<style>{load_css(CSS_PATH, os.path.getmtime(CSS_PATH))}</style>", unsafe_allow_html=True)

# Dark mode by default. .streamlit/config.toml defines a light and a dark theme, and Streamlit
# remembers the visitor's choice (app menu → Settings) in the browser under "stActiveTheme-…".
# On a visitor's first load we switch that to "Dark" and reload once, marking it done with our
# own "cyberDarkDefault" key; after that the visitor's own choice (Light / Dark / System) is kept.
st.html("""<script>(() => {
  try {
    if (localStorage.getItem("cyberDarkDefault")) return;
    localStorage.setItem("cyberDarkDefault", "1");
    localStorage.setItem(`stActiveTheme-${window.location.pathname}-v2`, JSON.stringify("Dark"));
    location.reload();
  } catch (e) {}   /* storage blocked (private window): fall back to the system setting */
})();</script>""", unsafe_allow_javascript=True)

# Retrieval methods offered in the sidebar, with a one-line explanation each
METHODS = {
    "Similarity": "Top-k slides closest in meaning to your question. Fastest.",
    "MMR (diverse)": "Relevant slides that aren't near-duplicates of each other.",
    "Multi-query + RRF": "Searches 3 rephrasings of your question and fuses the results. Best recall.",
    "Hybrid (vector + BM25)": "Meaning + exact keywords. Best for acronyms like SCI, DRE, KLOC.",
}

# Example questions shown before the first message (clickable)
SUGGESTIONS = [
    "What is defect density and how is it calculated?",
    "What is a software configuration item?",
    "How is quality assurance different from quality control?",
    "Why do we measure software with metrics?",
]


@st.cache_resource(show_spinner="Loading the slide index…")
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

n_files = len({d.metadata.get("source") for d in all_docs})


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
    st.markdown(
        '<div class="brand"><div class="logo">L</div>'
        '<div><div class="name">Slides Assistant</div><div class="sub">BSE2201 · RAG</div></div></div>',
        unsafe_allow_html=True)

    st.markdown('<div class="label">Retrieval</div>', unsafe_allow_html=True)
    method = st.selectbox("Method", list(METHODS), label_visibility="collapsed")
    st.markdown(f'<div class="method-help">{METHODS[method]}</div>', unsafe_allow_html=True)
    k = st.slider("Slides per answer (k)", 1, 10, 4)   # how many chunks go into the prompt

    st.markdown('<div class="label">Knowledge base</div>', unsafe_allow_html=True)
    embed_name = LOCAL_EMBEDDING_MODEL.split("/")[-1] if EMBEDDING_PROVIDER == "local" else "gemini-embedding-001"
    st.markdown(
        '<div class="glass side-card">'
        f'<div class="row"><span>Lecture files</span><b>{n_files}</b></div>'
        f'<div class="row"><span>Indexed chunks</span><b>{len(all_docs)}</b></div>'
        f'<div class="row"><span>Embeddings</span><b>{html.escape(embed_name)}</b></div>'
        f'<div class="row"><span>Answer model</span><b>{html.escape(CHAT_MODELS[0])}</b></div>'
        '</div>', unsafe_allow_html=True)

    if st.button("New chat", icon=":material/refresh:", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# ---------- Header ----------

# Inline SVG icons (Lucide-style line icons) for the hero chips; they inherit the chip's text colour
_SVG = ('<svg class="ico" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{}</svg>')
ICONS = {
    "book": _SVG.format('<path d="M2 4h6a4 4 0 0 1 4 4v12a3 3 0 0 0-3-3H2z"/><path d="M22 4h-6a4 4 0 0 0-4 4v12a3 3 0 0 1 3-3h7z"/>'),
    "search": _SVG.format('<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>'),
    "layers": _SVG.format('<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 12 10 5 10-5"/><path d="m2 17 10 5 10-5"/>'),
}

st.markdown(
    '<div class="glass hero">'
    '<div class="eyebrow">Lecture Slides Assistant</div>'
    '<h1>Ask your BSE2201 slides anything</h1>'
    '<p>Answers come only from the lecture material, with the exact slides they were taken from.</p>'
    '<div class="chips">'
    '<span class="chip"><span class="dot"></span>Online</span>'
    f'<span class="chip">{ICONS["book"]}{n_files} units · {len(all_docs)} chunks</span>'
    f'<span class="chip">{ICONS["search"]}{html.escape(method)}</span>'
    f'<span class="chip">{ICONS["layers"]}k = {k}</span>'
    '</div></div>', unsafe_allow_html=True)


# ---------- Chat rendering helpers ----------

AVATARS = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}


def chat_bubble(role, i):
    """A chat message wrapped in a keyed container, so the CSS can style user and assistant
    turns differently via the generated .st-key-msg-user-<i> / .st-key-msg-assistant-<i> class."""
    with st.container(key=f"msg-{role}-{i}"):
        return st.chat_message(role, avatar=AVATARS[role])


def short_label(label):
    """'UNIT 3.pdf, slide 25' → 'UNIT 3 · slide 25' for the source chips."""
    return label.replace(".pdf,", " ·")


def show_sources(sources, search=None):
    """Rewritten-search note, source chips, and a collapsible list of slide excerpts."""
    if search:
        st.markdown(f'<div class="searched">Searched for: <b>{html.escape(search)}</b></div>',
                    unsafe_allow_html=True)
    if not sources:
        return
    chips = "".join(f'<span class="src">{html.escape(short_label(label))}</span>' for label, _ in sources)
    st.markdown(f'<div class="sources">{chips}</div>', unsafe_allow_html=True)
    with st.expander(f"View {len(sources)} slide excerpts"):
        for i, (label, text) in enumerate(sources, 1):
            st.markdown(f"**{i}. {label}**")
            st.text(text[:800])


# The chat so far, kept across reruns: list of dicts {role, content, sources, search}
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-draw the previous messages (Streamlit redraws the whole page on every run)
for i, m in enumerate(st.session_state.messages):
    with chat_bubble(m["role"], i):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_sources(m.get("sources"), m.get("search"))

# Empty state: clickable example questions
clicked = None
if not st.session_state.messages:
    st.markdown('<div class="label">Try one of these</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, key=f"suggest-{i}", width="stretch"):
            clicked = s


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


# A question comes from the chat box or from a clicked suggestion
question = st.chat_input("Ask about your lecture units…") or clicked
if question:
    history = history_as_messages()                 # history BEFORE this question
    st.session_state.messages.append({"role": "user", "content": question})
    with chat_bubble("user", len(st.session_state.messages) - 1):
        st.markdown(question)

    with chat_bubble("assistant", len(st.session_state.messages)):
        try:
            with st.spinner("Searching your slides…"):
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
            answer = quota_message(e) or f":material/error: Error: {str(e)[:500]}"
            docs, search_question = [], None
            st.markdown(answer)

        # Show what was actually searched (if rewritten) and which slides were used
        sources = [(describe(d), d.page_content) for d in docs]
        rewritten = search_question if search_question and search_question != question else None
        show_sources(sources, rewritten)

        # Save the assistant turn so it is redrawn on the next run and used as history
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "search": rewritten})
        if clicked: #redraw without the suggestion button  
            st.rerun()
