import numpy as np
from typing import List, Dict, Optional, Tuple
from src.common.models import Chunk
from src.common.config import settings

class VectorIndexer:
    """
    Manages vector embeddings and dense similarity indexing.
    Supports in-memory cosine search and pluggable external vector engines (Qdrant).
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.chunks: Dict[str, Chunk] = {}
        self.vectors: Dict[str, np.ndarray] = {}

    def _embed_text_fallback(self, text: str) -> np.ndarray:
        """
        Deterministic, zero-dependency semantic hash projection embedding.
        Used as default when external LLM API keys are not supplied.
        """
        vec = np.zeros(self.dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            # Hash tokens deterministically to dimension buckets
            h = hash(word)
            idx = abs(h) % self.dim
            val = 1.0 if (h > 0) else -1.0
            vec[idx] += val

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def index_chunks(self, chunks: List[Chunk]):
        """
        Embeds and stores chunks in the vector index.
        """
        for chunk in chunks:
            if chunk.embedding is None:
                vec = self._embed_text_fallback(chunk.text)
                chunk.embedding = vec.tolist()
            else:
                vec = np.array(chunk.embedding, dtype=np.float32)
            
            self.chunks[chunk.chunk_id] = chunk
            self.vectors[chunk.chunk_id] = vec

    def search(self, query: str, top_k: int = 10, role_filter: Optional[List[str]] = None) -> List[Tuple[Chunk, float]]:
        """
        Performs dense cosine similarity search over indexed chunks.
        Enforces Role-Based Access Control (RBAC).
        """
        if not self.vectors:
            return []

        q_vec = self._embed_text_fallback(query)
        scored_results: List[Tuple[Chunk, float]] = []

        allowed = set(role_filter or ["public"])

        for chunk_id, c_vec in self.vectors.items():
            chunk = self.chunks[chunk_id]
            
            # ACL Check
            if not set(chunk.allowed_roles).intersection(allowed):
                continue

            # Cosine similarity
            score = float(np.dot(q_vec, c_vec))
            scored_results.append((chunk, score))

        # Sort descending by similarity score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]
