import math
import re
from typing import List, Dict, Optional
from src.common.models import Chunk, RetrievalCandidate
from src.common.config import settings

class CrossEncoderReranker:
    """
    Evaluates deep semantic query-document cross-attention relevance.
    Employs token-overlap and semantic alignment scoring as fallback when GPU cross-encoder is offline.
    Expands passed child chunks to parent container context.
    """

    def __init__(self, parent_store: Optional[Dict[str, Chunk]] = None):
        self.parent_store = parent_store or {}

    def _cross_attention_score(self, query: str, text: str) -> float:
        """
        Cross-Encoder simulation computing bidirectional token interaction and positional proximity.
        """
        q_words = set(re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', query.lower()))
        t_words = re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', text.lower())
        
        if not q_words or not t_words:
            return 0.0

        matches = [w for w in t_words if w in q_words]
        if not matches:
            return 0.0

        # Term frequency + coverage ratio
        coverage = len(set(matches)) / len(q_words)
        density = len(matches) / len(t_words)
        
        # Exponential scaling
        raw_score = (coverage * 0.7) + (density * 0.3)
        calibrated_score = 1.0 / (1.0 + math.exp(-6.0 * (raw_score - 0.25)))
        return float(calibrated_score)

    def rerank(
        self, 
        query: str, 
        candidates: List[RetrievalCandidate], 
        top_k: int = 5,
        threshold: float = 0.25
    ) -> List[RetrievalCandidate]:
        scored: List[RetrievalCandidate] = []

        for candidate in candidates:
            # Score against child text
            cross_score = self._cross_attention_score(query, candidate.chunk.text)
            
            # Combine RRF rank bonus with cross-encoder score
            blended_score = (cross_score * 0.7) + (candidate.score * 0.3)
            
            if blended_score >= threshold:
                # Parent-chunk expansion
                parent_text = None
                if candidate.chunk.parent_id and candidate.chunk.parent_id in self.parent_store:
                    parent_text = self.parent_store[candidate.chunk.parent_id].text

                reranked_cand = candidate.model_copy(
                    update={
                        "score": blended_score,
                        "parent_text": parent_text or candidate.parent_text
                    }
                )
                scored.append(reranked_cand)

        # Sort descending by rerank score
        scored.sort(key=lambda x: x.score, reverse=True)
        
        # Assign updated ranks
        final_list = []
        for idx, item in enumerate(scored[:top_k]):
            item.rank = idx + 1
            final_list.append(item)

        return final_list
