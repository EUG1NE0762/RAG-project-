"""
STEP 4 — History-aware generation: a terminal chat that understands follow-up questions.

Problem: a follow-up like "Give me an example of it" can't be searched on its own — "it" has
no meaning without the earlier conversation. Solution (query rewriting / condensing):
    1. ask the LLM to rewrite the follow-up into a standalone question using the chat history
       ("Give me an example of it" → "Provide an example calculation of defect density.")
    2. retrieve with the rewritten question
    3. answer with the retrieved chunks + the chat history
    4. remember the question and answer for next time

Run:  venv\\Scripts\\python.exe 4_history_aware_generation.py   (type 'quit' to exit)
"""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from gemini_setup import get_chat_model, get_vector_store, describe

load_dotenv()

# Connect to the slide database and the chat model
db = get_vector_store()
model = get_chat_model()

# The conversation so far, as alternating HumanMessage (user) / AIMessage (assistant) objects
chat_history = []


def ask_question(user_question):
    """Answer one question, using and then updating chat_history."""
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Make the question standalone using conversation history (skipped for the first question)
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Find relevant chunks with the (possibly rewritten) question
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        preview = '\n'.join(doc.page_content.split('\n')[:2])   # first 2 lines of each chunk
        print(f"  Doc {i} [{describe(doc)}]: {preview}...")

    # Step 3: Create the final prompt (the user's ORIGINAL wording + the retrieved chunks)
    documents = "\n".join(f"- [{describe(doc)}] {doc.page_content}" for doc in docs)
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

Documents:
{documents}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

    # Step 4: Get the answer — chat history is included so the model knows what "it" refers to
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    answer = model.invoke(messages).content

    # Step 5: Remember this exchange for the next question
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer


def start_chat():
    """Simple read-answer loop in the terminal."""
    print("Ask me questions about the lecture slides! Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            print("Goodbye!")
            break
        ask_question(question)


if __name__ == "__main__":
    start_chat()
