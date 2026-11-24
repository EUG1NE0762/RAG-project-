"""
STEP 3 — Answer generation: Retrieval + Generation = RAG.

    question ──retrieve──▶ top-k slide chunks ──put into prompt──▶ Gemini ──▶ grounded answer

The model is told to answer ONLY from the retrieved chunks ("grounding"), which keeps the
answer tied to the lecture material and reduces hallucination (made-up facts).

Run:  venv\\Scripts\\python.exe 3_answer_generation.py
"""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from gemini_setup import get_vector_store, get_chat_model, describe

load_dotenv()

# Open the vector store built by 1_ingestion_pipeline.py
db = get_vector_store()

# The question to answer
query = "What is defect density and how is it calculated?"

# ── Retrieval: the 5 most similar chunks ──
retriever = db.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i} [{describe(doc)}]:\n{doc.page_content}\n")

# ── Augmentation: build a prompt that contains the question AND the retrieved chunks ──
# Each chunk is labelled with its file and slide so the model can cite where facts came from.
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- [{describe(doc)}] {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# ── Generation: Gemini writes the answer (falls back across CHAT_MODELS if one is unavailable) ──
model = get_chat_model()

# SystemMessage sets the assistant's role; HumanMessage carries the actual request
messages = [
    SystemMessage(content="You are a helpful study assistant for lecture slides."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)   # returns an AIMessage; the text is in .content
                                     # returns an AIMessage; the text is in .content

print("\n--- Generated Response ---")
print(result.content)
