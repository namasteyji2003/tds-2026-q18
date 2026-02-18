import numpy as np
import hashlib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    docs: list[str]
    query: str

def get_embedding(text: str):
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)
    np.random.seed(seed)
    return np.random.rand(384)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.post("/search")
def search(payload: SearchRequest):

    docs = payload.docs
    query = payload.query

    doc_embeddings = [get_embedding(doc) for doc in docs]
    query_embedding = get_embedding(query)

    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]

    top_indices = np.argsort(similarities)[::-1][:3]
    matches = [docs[i] for i in top_indices]

    return {"matches": matches}
