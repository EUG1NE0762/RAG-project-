import time
import os, glob, json
import numpy as np
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types

GEN_MODEL = "gemini-3.8-flash"        # check AI Studio for current model names
EMBED_MODEL = "gemini-embedding-001"
VEC_FILE, META_FILE = "index_vectors.npy", "index_meta.json"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def embed(texts, task, batch=90):
    vectors = []
    for i in range(0, len(texts), batch):
        for attempt in range(5):
            try:
                res = client.models.embed_content(
                    model=EMBED_MODEL, contents=texts[i:i+batch],
                    config=types.EmbedContentConfig(task_type=task))
                vectors += [e.values for e in res.embeddings]
                break
            except Exception as e:
                if "429" in str(e):
                    time.sleep(60)   # quota hit: wait a minute, then retry
                else:
                    raise
        else:
            raise RuntimeError("Embedding failed after 5 retries")
        if i + batch < len(texts):
            time.sleep(62)           # stay under 100 requests per minute
    v = np.array(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def ingest():
    texts, metas = [], []
    for path in glob.glob("slides/*.pdf"):
        name = os.path.basename(path)
        for num, page in enumerate(PdfReader(path).pages, 1):
            text = (page.extract_text() or "").strip()
            if len(text) >= 20:
                texts.append(text)
                metas.append({"file": name, "slide": num, "text": text})
    np.save(VEC_FILE, embed(texts, "RETRIEVAL_DOCUMENT"))
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metas, f)
    load_index.clear()
    return len(texts)

@st.cache_data
def load_index():
    if not (os.path.exists(VEC_FILE) and os.path.exists(META_FILE)):
        return None, None
    with open(META_FILE, encoding="utf-8") as f:
        return np.load(VEC_FILE), json.load(f)
def generate(prompt, tries=5):
    for attempt in range(tries):
        try:
            return client.models.generate_content(model=GEN_MODEL, contents=prompt).text
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < tries - 1:
                time.sleep(3 * (attempt + 1)) #This will make it wait 3, 6, 9, 12s then retry
            else:
                raise

def answer(question, k=4):
    vecs, metas = load_index()
    q = embed([question], "RETRIEVAL_QUERY")[0]
    top = np.argsort(vecs @ q)[::-1][:k]          # retrieval step
    hits = [metas[i] for i in top]
    context = "\n---\n".join(f"[{h['file']}, slide {h['slide']}]\n{h['text']}" for h in hits)
    prompt = ("You are a study assistant. Answer using ONLY the lecture content below. "
              "If the answer is not there, say you don't know.\n\n"
              f"Content:\n{context}\n\nQuestion: {question}")
    return generate(prompt), hits

st.title("Lecture Slides Assistant")
if st.sidebar.button("Index slides"):
    with st.spinner("Indexing..."):
        n = ingest()
    st.sidebar.success(f"{n} slides indexed")

q = st.chat_input("Ask about your lecture units")
if q:
    st.chat_message("user").write(q)
    if load_index()[0] is None:
        st.warning("Click 'Index slides' in the sidebar first.")
    else:
        with st.spinner("Searching slides..."):
            try:
                text, sources = answer(q)
            except Exception:
                st.error("Gemini is busy right now. Please try again in a few seconds.")
                st.stop()