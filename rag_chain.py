# RAG logic using Groq (Anthropic and Gemini need credits)
import os

from groq import Groq
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)


def get_retriever():
    """Creates a connection to your existing ChromaDB (the one ingest.py already filled).
    It loads the same embedding model so it can convert the user's question into a vector
    and find the most similar chunks. k=3 means \"fetch the 3 most relevant chunks.\""""
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return db.as_retriever(search_kwargs={"k": 3})


def ask(question: str) -> str:
    """This is the heart of RAG"""
    retriever = get_retriever()

    # Retrieve relevant chunks from ChromaDB
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build prompt with retrieved context
    prompt = f"""You are a supply chain assistant for Apple EMEIA operations.
Use only the context below to answer the question. 
If the answer isn't in the context, say "I don't have that information."

Context:
"{context}"

Question: "{question}"

Answer:"""

    # Call Groq
    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}  # role: "user" — the human speaking; "assistant" — Claude's response
        ]
    )
    return message.choices[0].message.content


if __name__ == "__main__":
    """This gets run only if I'm executing this file directly, not importing it."""
    print(ask("What are the main delay risks in EMEIA logistics?"))
