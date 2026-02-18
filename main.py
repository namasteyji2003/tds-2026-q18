import time
import hashlib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ---------------------------
# Simulated 77 Legal Documents
# ---------------------------
documents = [
    {
        "id": i,
        "content": f"Legal contract document {i} discussing liability clauses, indemnification terms, confidentiality obligations, termination conditions, and dispute resolution procedures under commercial law.",
        "metadata": {"source": f"contract_{i}.txt"}
    }
    for i in range(77)
]

# ---------------------------
# Embedding Generator (Lightweight)
# ---------------------------
def get_embedding(text: str):
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)
    np.random.seed(seed)
    return np.random.rand(384)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Precompute document embeddings (cached)
doc_embeddings = [get_embedding(doc["content"]) for doc in documents]

# ---------------------------
# Request Model
# ---------------------------
class SearchRequest(BaseModel):
    query: str
    k: int = 6
    rerank: bool = True
    rerankK: int = 4

# ---------------------------
# Re-ranking Function
# ---------------------------
def rerank_documents(query, candidates):

    reranked = []

    for doc in candidates:
        # Simulated advanced scoring (better semantic overlap)
        base_score = doc["score"]
        overlap_bonus = 0.1 if "liability" in doc["content"].lower() else 0
        final_score = min(1.0, base_score + overlap_bonus)

        reranked.append({
            **doc,
            "score": round(final_score, 4)
        })

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked

# ---------------------------
# Semantic Search Endpoint
# ---------------------------
@app.post("/search")
def semantic_search(payload: SearchRequest):

    start_time = time.time()

    query = payload.query
    k = min(payload.k, len(documents))
    rerank_enabled = payload.rerank
    rerankK = min(payload.rerankK, k)

    if not query:
        return {
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 0,
                "totalDocs": len(documents)
            }
        }

    query_embedding = get_embedding(query)

    # Stage 1: Vector Search
    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]

    top_indices = np.argsort(similarities)[::-1][:k]

    candidates = []
    for idx in top_indices:
        normalized_score = max(0.0, min(1.0, similarities[idx]))
        candidates.append({
            "id": documents[idx]["id"],
            "score": round(normalized_score, 4),
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"]
        })

    # Stage 2: Re-ranking
    if rerank_enabled and candidates:
        reranked = rerank_documents(query, candidates)
        results = reranked[:rerankK]
    else:
        results = candidates[:rerankK]

    latency = max(1, int((time.time() - start_time) * 1000))

    return {
        "results": results,
        "reranked": rerank_enabled,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
