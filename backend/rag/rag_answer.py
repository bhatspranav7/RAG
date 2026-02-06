import ollama
import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# LOAD EMBEDDING MODEL ONCE (VERY IMPORTANT)
# --------------------------------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def retrieve(query, index, chunks, k=5):
    """
    Convert query to embedding and search FAISS.
    """

    query_vector = embed_model.encode([query]).astype("float32")

    distances, ids = index.search(query_vector, k)

    results = []
    for i in ids[0]:
        if i != -1:
            results.append(chunks[i])

    return results


# --------------------------------------------------
# GENERATE ANSWER USING OLLAMA
# --------------------------------------------------

def generate_answer(question, retrieved_chunks):
    """
    Send grounded prompt to local LLM.
    """

    if not retrieved_chunks:
        return "I don't have enough information in the knowledge base."

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a professional insurance assistant.

STRICT RULES:
- Answer ONLY using the provided context.
- If missing, say you don't have that information.
- Do NOT invent facts.
- Keep answers under 4 sentences.
- Speak in a helpful, human tone.


CONTEXT:
{context}

QUESTION:
{question}
"""

    try:
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}]
)


        return response["message"]["content"]

    except Exception as e:

        print("\n🚨 OLLAMA CONNECTION ERROR:")
        print(str(e))

        return """
LLM connection failed.

Make sure:

1. Ollama is running on Windows
2. Model llama3.2 is installed
3. host.docker.internal is reachable from WSL
"""
