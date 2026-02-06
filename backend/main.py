import os
from fastapi import FastAPI
from pydantic import BaseModel

from rag.make_sample_pdf import make_pdf
from rag.pdf_to_text import pdf_to_text
from rag.chunking import chunk_text
from rag.embed_store import build_index, load_index
from rag.rag_answer import retrieve, generate_answer


app = FastAPI(title="Local RAG API")


# -----------------------------
# PATH SETUP (VERY IMPORTANT)
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_PATH = os.path.join(DATA_DIR, "knowledge.pdf")
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")


# -----------------------------
# GLOBAL CACHE (LAZY LOADING)
# -----------------------------

index = None
chunks = None


# -----------------------------
# REQUEST MODEL
# -----------------------------

class ChatRequest(BaseModel):
    message: str


# -----------------------------
# INGEST ENDPOINT
# -----------------------------

@app.post("/ingest")
def ingest():
    """
    Build vector index from PDF.
    Run this ONCE unless data changes.
    """

    global index, chunks

    os.makedirs(DATA_DIR, exist_ok=True)

    # Create sample PDF if missing
    if not os.path.exists(PDF_PATH):
        print("Creating sample knowledge base...")
        make_pdf(PDF_PATH)

    print("Extracting text from PDF...")
    text = pdf_to_text(PDF_PATH)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Building FAISS index (may take ~10-20s first time)...")
    build_index(chunks, INDEX_PATH)

    print("Loading index into memory...")
    index, chunks = load_index(INDEX_PATH)

    return {
        "status": "success",
        "chunks_indexed": len(chunks)
    }


# -----------------------------
# CHAT ENDPOINT
# -----------------------------

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Query the knowledge base.
    """

    global index, chunks

    # 🔥 LAZY LOAD — prevents rebuild on every request
    if index is None or chunks is None:

        if not os.path.exists(INDEX_PATH):
            return {
                "answer": "Knowledge base not ready. Call /ingest first."
            }

        print("Lazy loading FAISS index...")
        index, chunks = load_index(INDEX_PATH)

    # Retrieve context
    hits = retrieve(req.message, index, chunks)

    # Generate answer via Ollama
    answer = generate_answer(req.message, hits)

    return {
        "answer": answer,
        "sources": hits  # optional but GREAT for debugging
    }


# -----------------------------
# HEALTH CHECK (PRO LEVEL)
# -----------------------------

@app.get("/")
def root():
    return {
        "status": "RAG API running"
    }
