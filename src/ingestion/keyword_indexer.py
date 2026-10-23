import re
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from src.common.models import Chunk

class KeywordIndexer:
    """
    Manages sparse keyword inverted index using BM25Okapi algorithm.
    Optimized for exact keyword matches, identifiers, abbreviations, and codes.
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_id_map: Dict[int, str] = {}

    def _tokenize(self, text: str) -> List[str]:
        # Clean and split into lowercased alphanumeric tokens
        tokens = re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', text.lower())
        return tokens

    def index_chunks(self, chunks: List[Chunk]):
        self.chunks.extend(chunks)
        start_idx = len(self.corpus_tokens)
        
        for idx, chunk in enumerate(chunks):
            tokens = self._tokenize(chunk.text)
            self.corpus_tokens.append(tokens)
            self.chunk_id_map[start_idx + idx] = chunk.chunk_id

        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 10, role_filter: Optional[List[str]] = None) -> List[Tuple[Chunk, float]]:
        if not self.bm25 or not self.chunks:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)
        allowed = set(role_filter or ["public"])

        results: List[Tuple[Chunk, float]] = []
        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue
            chunk = self.chunks[idx]
            
            # ACL Check
            if not set(chunk.allowed_roles).intersection(allowed):
                continue
                
            results.append((chunk, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
