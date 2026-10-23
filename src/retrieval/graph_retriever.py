import asyncio
from typing import List, Dict, Optional
from src.common.models import Chunk, RetrievalCandidate
from src.ingestion.graph_indexer import KnowledgeGraphIndexer

class GraphRetriever:
    """
    Performs Knowledge Graph traversal based on extracted query entities.
    Discovers multi-hop relational paths, connected nodes, and source chunks.
    """

    def __init__(self, graph_indexer: KnowledgeGraphIndexer, chunk_store: Dict[str, Chunk]):
        self.graph_indexer = graph_indexer
        self.chunk_store = chunk_store

    async def search(
        self, 
        entities: List[str], 
        top_k: int = 10,
        role_filter: Optional[List[str]] = None
    ) -> List[RetrievalCandidate]:
        if not entities:
            return []

        loop = asyncio.get_event_loop()
        subgraph_data = await loop.run_in_executor(
            None, 
            self.graph_indexer.query_subgraph, 
            entities, 
            2 # 2-hop traversal
        )

        candidates: List[RetrievalCandidate] = []
        allowed = set(role_filter or ["public"])
        paths_summary = "; ".join(subgraph_data.get("paths", [])[:5])

        for idx, chunk_id in enumerate(subgraph_data.get("chunk_ids", [])):
            chunk = self.chunk_store.get(chunk_id)
            if not chunk:
                continue

            # ACL Check
            if not set(chunk.allowed_roles).intersection(allowed):
                continue

            # Inject graph path summary into metadata
            enriched_metadata = {
                **chunk.metadata,
                "graph_paths": paths_summary,
                "matched_entities": subgraph_data.get("entities", [])
            }
            enriched_chunk = chunk.model_copy(update={"metadata": enriched_metadata})

            candidates.append(
                RetrievalCandidate(
                    chunk=enriched_chunk,
                    source_strategy="knowledge_graph",
                    score=1.0 / (idx + 1), # Graph distance rank heuristic
                    rank=idx + 1
                )
            )
            if len(candidates) >= top_k:
                break

        return candidates
