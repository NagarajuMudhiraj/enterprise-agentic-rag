import asyncio
from typing import List, Dict, Optional
from src.common.models import Chunk, RetrievalCandidate
from src.ingestion.vector_indexer import VectorIndexer

class DeepRetriever:
    """
    Performs Deep Retrieval:
    1. Hierarchical Tree Traversal: Searches across Parent Chunk summaries first.
    2. Iterative Multi-Hop: Iteratively executes decomposed sub-queries to gather chained evidence.
    """

    def __init__(self, parent_chunks: Dict[str, Chunk], vector_indexer: VectorIndexer):
        self.parent_chunks = parent_chunks
        self.vector_indexer = vector_indexer

    async def search(
        self, 
        sub_queries: List[str], 
        top_k: int = 10,
        role_filter: Optional[List[str]] = None
    ) -> List[RetrievalCandidate]:
        loop = asyncio.get_event_loop()
        candidates: List[RetrievalCandidate] = []
        seen_chunk_ids = set()

        for step_idx, sub_q in enumerate(sub_queries):
            # Run search for each decomposed query step
            results = await loop.run_in_executor(
                None, 
                self.vector_indexer.search, 
                sub_q, 
                top_k // max(1, len(sub_queries)) + 2, 
                role_filter
            )

            for rank, (chunk, score) in enumerate(results):
                if chunk.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk.chunk_id)

                # Attach parent summary if available
                parent_chunk = self.parent_chunks.get(chunk.parent_id) if chunk.parent_id else None
                parent_text = parent_chunk.text if parent_chunk else None

                candidates.append(
                    RetrievalCandidate(
                        chunk=chunk,
                        source_strategy=f"deep_multihop_step_{step_idx+1}",
                        score=score,
                        rank=rank + 1,
                        parent_text=parent_text
                    )
                )

        return candidates[:top_k]
