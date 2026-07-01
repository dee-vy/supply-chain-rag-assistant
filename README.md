# Supply Chain RAG Assistant

An AI-powered Q&A system for supply chain operations, built using 
Retrieval-Augmented Generation (RAG). The system answers questions 
grounded in domain-specific supply chain documents, preventing 
hallucination by restricting responses to retrieved context only.

## Architecture
User Question → FastAPI → RAG Chain → ChromaDB (semantic search)
→ LLM (Groq/LLaMA 3.3) → Answer

## Tech Stack

- **LLM**: LLaMA 3.3 70B via Groq API
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) — runs locally
- **Vector Store**: ChromaDB — local persistent vector database
- **RAG Framework**: LangChain
- **API**: FastAPI + Uvicorn
- **Language**: Python 3.9+

## Features

- Domain-grounded answers — LLM only responds based on ingested documents
- Hallucination prevention — returns "I don't have that information" 
  for out-of-context questions
- Semantic search — finds relevant context by meaning, not just keywords
- REST API — production-ready endpoint with request validation
- Auto-generated API docs via FastAPI at `/docs`

## Project Structure
rag_supply_chain/
```
├── data/
│   └── supply_chain.txt     # Domain knowledge base
├── chroma_db/               # Persisted vector store (auto-generated)
├── ingest.py                # Document ingestion and embedding pipeline
├── rag_chain.py             # RAG logic — retrieval + LLM call
├── controller.py            # FastAPI app — REST endpoints
└── README.md
```

## Setup

1. Clone the repo and create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install langchain langchain-community chromadb \
            sentence-transformers fastapi uvicorn groq
```

3. Set your Groq API key:
```bash
export GROQ_API_KEY="your_key_here"
```

4. Ingest documents into ChromaDB:
```bash
python ingest.py
```

5. Start the API server:
```bash
python controller.py
```

6. Visit **http://localhost:8000/docs** to test the API interactively.

## API Usage

**POST /ask**
```json
{
  "question": "What are the main delay risks in EMEIA logistics?"
}
```

**Response**
```json
{
  "question": "What are the main delay risks in EMEIA logistics?",
  "answer": "The main delay risks include port congestion at Rotterdam, customs clearance delays in Germany, and seasonal demand spikes in Q4."
}
```

## How RAG Works

1. **Ingest** - documents are split into chunks and converted to 
   embedding vectors, stored in ChromaDB
2. **Retrieve** - user question is embedded and semantically searched 
   against ChromaDB, returning the top 3 most relevant chunks
3. **Generate** - retrieved chunks are injected into a prompt as context, 
   sent to the LLM which generates a grounded answer