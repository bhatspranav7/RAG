from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks):
    vectors = model.encode(chunks)
    return np.array(vectors).astype("float32")


def build_index(chunks, index_path):
    vectors = embed_chunks(chunks)

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, index_path)

    with open(index_path + ".pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_index(index_path):
    index = faiss.read_index(index_path)

    with open(index_path + ".pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks
