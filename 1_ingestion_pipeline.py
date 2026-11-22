"""
STEP 1 — Ingestion pipeline: turn the lecture PDFs into a searchable vector database.

    slides/*.pdf ──load──▶ 1 Document per slide ──split──▶ chunks ──embed──▶ Chroma (db/)

Run once (and again whenever the PDFs in slides/ change):
    venv\\Scripts\\python.exe 1_ingestion_pipeline.py
"""
import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader   # reads every PDF in a folder
from langchain_text_splitters import RecursiveCharacterTextSplitter     # splits long text into chunks
from langchain_chroma import Chroma                                     # the vector database
from dotenv import load_dotenv
from gemini_setup import get_embeddings, SLIDES_PATH, PERSIST_DIR, EMBEDDING_PROVIDER

load_dotenv()


def load_documents(docs_path=SLIDES_PATH):
    """Load every PDF in the slides directory — one LangChain Document per slide (page).

    A Document has .page_content (the slide's text) and .metadata
    ({'source': 'slides/UNIT 1.pdf', 'page': 0, ...}), used later to cite sources.
    """
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your lecture PDFs.")

    loader = PyPDFDirectoryLoader(docs_path)
    documents = loader.load()

    # Drop near-empty pages (title-only or image-only slides) — they add noise, not answers
    documents = [d for d in documents if len(d.page_content.strip()) >= 20]

    if len(documents) == 0:
        raise FileNotFoundError(f"No readable PDF pages found in {docs_path}. Please add your lecture slides.")

    print(f"Loaded {len(documents)} slides")
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents as a sanity check
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into chunks of at most `chunk_size` characters.

    Chunking keeps each piece small and focused, so a search returns just the relevant part
    and the prompt sent to the LLM stays short. RecursiveCharacterTextSplitter (rag-for-beginners
    step 5) tries the separators in order — paragraph, line, sentence, word — and only falls back
    to the next one when a piece is still too long, so chunks rarely cut a sentence in half.
    Most slides are under 1000 characters, so most become a single chunk.
    """
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],   # tried left to right
        chunk_size=chunk_size,                      # max characters per chunk
        chunk_overlap=chunk_overlap                 # characters repeated between neighbouring chunks
    )

    chunks = text_splitter.split_documents(documents)   # metadata (source, page) is copied to each chunk

    if chunks:
        for i, chunk in enumerate(chunks[:5]):          # preview the first 5 chunks
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']} (page {chunk.metadata.get('page', 0) + 1})")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


def chunk_ids(chunks):
    """Give every chunk a stable id like 'UNIT 1.pdf-p0-c1' (file, page, chunk number on that page).

    The same chunk always gets the same id, so re-running the script can skip chunks that
    are already stored and resume where a previous run stopped.
    """
    ids, seen = [], {}
    for c in chunks:
        base = f"{os.path.basename(c.metadata['source'])}-p{c.metadata.get('page', 0)}"
        seen[base] = seen.get(base, 0) + 1          # count chunks per page: c1, c2, ...
        ids.append(f"{base}-c{seen[base]}")
    return ids


def create_vector_store(chunks, persist_directory=PERSIST_DIR, batch_size=40, pause_seconds=65):
    """Embed the chunks and save them in a persistent Chroma vector store.

    Embedding = converting each chunk's text into a vector (list of numbers) that captures its
    meaning; similar meanings give nearby vectors. Chroma stores the vectors + text + metadata
    on disk (persist_directory) so later steps can search without re-embedding.

    For the Gemini free tier (100 embedding requests/minute) chunks are sent in batches with a
    pause in between; the local model has no limits, so main() passes a big batch and no pause.
    """
    print(f"Creating embeddings ({EMBEDDING_PROVIDER}) and storing in ChromaDB...")

    embedding_model = get_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}    # compare vectors by cosine similarity
    )

    # Only embed chunks whose id isn't stored yet (makes the script resumable)
    ids = chunk_ids(chunks)
    existing = set(vectorstore.get(include=[])["ids"])
    todo = [(i, c) for i, c in zip(ids, chunks) if i not in existing]
    print(f"--- {len(existing)} chunks already stored, {len(todo)} to embed ---")

    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        for attempt in range(5):                       # up to 5 tries per batch
            try:
                vectorstore.add_documents([c for _, c in batch], ids=[i for i, _ in batch])
                break
            except Exception as e:
                msg = str(e)
                if "PerDay" in msg:                    # daily limit: waiting won't help, stop now
                    raise RuntimeError("Daily Gemini embedding quota reached. Run this script again "
                                       "tomorrow (or with a paid key) and it will resume.") from e
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    print(f"  Rate limited, waiting {pause_seconds}s and retrying...")   # per-minute limit
                    time.sleep(pause_seconds)
                else:
                    raise
        else:                                          # for-else: runs only if no attempt succeeded
            raise RuntimeError("Still rate limited after several retries.")
        done = len(existing) + start + len(batch)
        print(f"  Stored {done}/{len(chunks)} chunks")
        if start + batch_size < len(todo):             # pause between batches (0 for local)
            time.sleep(pause_seconds)

    print("--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    """Main ingestion pipeline: load → split → embed & store."""
    print("=== RAG Document Ingestion Pipeline ===\n")

    # Step 1: Load documents (one per slide)
    documents = load_documents(SLIDES_PATH)

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Create vector store
    if EMBEDDING_PROVIDER == "gemini":
        vectorstore = create_vector_store(chunks, PERSIST_DIR)   # batched + paused for the free tier
    else:
        vectorstore = create_vector_store(chunks, PERSIST_DIR, batch_size=256, pause_seconds=0)  # local: no limits

    print("\n✅ Ingestion complete! Your slides are now ready for RAG queries.")
    return vectorstore


if __name__ == "__main__":   # run main() only when executed directly, not when imported
    main()
