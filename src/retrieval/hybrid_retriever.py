import asyncio
from typing import List, Optional
from src.common.models import Chunk, RetrievalCandidate
from src.ingestion.vector_indexer import VectorIndexer
from src.ingestion.keyword_indexer import KeywordIndexer

class HybridRetriever:
    """
    Executes concurrent Dense Vector + Sparse BM25 Keyword searches.
    """

    def __init__(self, vector_indexer: VectorIndexer, keyword_indexer: KeywordIndexer):
        self.vector_indexer = vector_indexer
        self.keyword_indexer = keyword_indexer

    async def search(
        self, 
        query: str, 
        top_k: int = 15, 
        role_filter: Optional[List[str]] = None,
        hyde_query: Optional[str] = None
    ) -> List[List[RetrievalCandidate]]:
        """
        Runs dense and sparse retrieval concurrently.
        Returns [dense_candidates, sparse_candidates]
        """
        search_query_dense = hyde_query if hyde_query else query
        
        async def run_dense():
            # Run in threadpool for non-blocking execution
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                self.vector_indexer.search, 
                search_query_dense, 
                top_k, 
                role_filter
            )
            return [
                RetrievalCandidate(
                    chunk=chunk,
                    source_strategy="dense_vector",
                    score=score,
                    rank=idx + 1
                )
                for idx, (chunk, score) in enumerate(results)
            ]

        async def run_sparse():
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                self.keyword_indexer.search, 
                query, 
                top_k, 
                role_filter
            )
            return [
                RetrievalCandidate(
                    chunk=chunk,
                    source_strategy="bm25_keyword",
                    score=score,
                    rank=idx + 1
                )
                for idx, (chunk, score) in enumerate(results)
            ]

        dense_cands, sparse_cands = await asyncio.gather(run_dense(), run_sparse())
        return [dense_cands, sparse_cands]
