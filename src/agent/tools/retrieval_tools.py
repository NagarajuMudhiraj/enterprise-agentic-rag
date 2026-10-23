from typing import List, Dict, Any, Optional
from src.agent.tools.registry import AgentTool, ToolRegistry
from src.ingestion.vector_indexer import VectorIndexer
from src.ingestion.keyword_indexer import KeywordIndexer
from src.ingestion.graph_indexer import KnowledgeGraphIndexer
from src.common.models import Chunk

def build_retrieval_tools(
    vector_indexer: VectorIndexer,
    keyword_indexer: KeywordIndexer,
    graph_indexer: KnowledgeGraphIndexer,
    chunk_store: Dict[str, Chunk],
    parent_store: Dict[str, Chunk],
) -> ToolRegistry:
    registry = ToolRegistry()

    # Tool 1: Hybrid Search
    async def run_hybrid_search(query: str, top_k: int = 5, role_filter: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        # Dense + Sparse
        v_results = vector_indexer.search(query, top_k=top_k, role_filter=role_filter)
        k_results = keyword_indexer.search(query, top_k=top_k, role_filter=role_filter)

        # Merge unique chunks
        seen = set()
        candidates = []
        for c, score in v_results + k_results:
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            tag = f"[{c.doc_id}#{c.chunk_id}]"
            candidates.append({
                "citation_tag": tag,
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "text": c.text,
                "score": round(score, 3)
            })
            if len(candidates) >= top_k:
                break

        return {
            "query": query,
            "results_count": len(candidates),
            "passages": candidates
        }


    registry.register(AgentTool(
        name="search_hybrid",
        description="Searches indexed documents using both semantic dense vector similarity and exact BM25 keyword matching. Best for factual lookups, policies, definitions, and code names.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific search terms or concept to retrieve from documents."
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of relevant evidence passages to return."
                }
            },
            "required": ["query"]
        },
        func=run_hybrid_search
    ))

    # Tool 2: Knowledge Graph Traversal
    async def run_graph_search(entity: str, max_hops: int = 2, role_filter: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        subgraph = graph_indexer.query_subgraph([entity], max_hops=max_hops)
        paths = subgraph.get("paths", [])
        connected_entities = subgraph.get("entities", [])
        matched_chunks = []
        allowed = set(role_filter or ["public"])
        
        for cid in subgraph.get("chunk_ids", [])[:5]:
            c = chunk_store.get(cid)
            if c and set(c.allowed_roles).intersection(allowed):
                matched_chunks.append({
                    "citation_tag": f"[{c.doc_id}#{c.chunk_id}]",
                    "text": c.text
                })


        return {
            "query_entity": entity,
            "connected_entities": connected_entities,
            "relationship_paths": paths,
            "source_chunks": matched_chunks
        }

    registry.register(AgentTool(
        name="query_knowledge_graph",
        description="Explores relationships, architecture dependencies, and connections for named entities in the Knowledge Graph. Best for answering 'What does X depend on?', 'How are X and Y connected?', or finding architectural topology.",
        parameters={
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "The exact name of the entity, service, database, or component to explore."
                },
                "max_hops": {
                    "type": "integer",
                    "default": 2,
                    "description": "Maximum traversal distance in the knowledge graph."
                }
            },
            "required": ["entity"]
        },
        func=run_graph_search
    ))

    # Tool 3: Read Parent Context
    async def run_read_parent(chunk_id: str, **kwargs) -> Dict[str, Any]:
        child = chunk_store.get(chunk_id)
        if not child:
            return {"status": "error", "message": f"Chunk '{chunk_id}' not found."}
        
        parent = parent_store.get(child.parent_id) if child.parent_id else None
        if not parent:
            return {
                "chunk_id": chunk_id,
                "text": child.text,
                "note": "No wider parent container found."
            }

        return {
            "chunk_id": chunk_id,
            "parent_id": parent.chunk_id,
            "full_section_text": parent.text,
            "section_title": parent.metadata.get("title", "Section")
        }

    registry.register(AgentTool(
        name="read_parent_context",
        description="Expands an isolated chunk to its complete parent section. Use when a retrieved chunk is clipped or requires surrounding paragraphs for context.",
        parameters={
            "type": "object",
            "properties": {
                "chunk_id": {
                    "type": "string",
                    "description": "The ID of the child chunk whose full section context is required."
                }
            },
            "required": ["chunk_id"]
        },
        func=run_read_parent
    ))

    # Tool 4: Compare Entities
    async def run_compare_entities(entity_a: str, entity_b: str, role_filter: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        sub_a = graph_indexer.query_subgraph([entity_a], max_hops=1)
        sub_b = graph_indexer.query_subgraph([entity_b], max_hops=1)

        v_a = vector_indexer.search(entity_a, top_k=2, role_filter=role_filter)
        v_b = vector_indexer.search(entity_b, top_k=2, role_filter=role_filter)

        return {
            "entity_a": {
                "name": entity_a,
                "paths": sub_a.get("paths", []),
                "passages": [c.text for c, _ in v_a]
            },
            "entity_b": {
                "name": entity_b,
                "paths": sub_b.get("paths", []),
                "passages": [c.text for c, _ in v_b]
            }
        }

    registry.register(AgentTool(
        name="compare_entities",
        description="Directly gathers comparative specifications, dependencies, and policies for two distinct components or policies side-by-side.",
        parameters={
            "type": "object",
            "properties": {
                "entity_a": {"type": "string", "description": "First entity or concept."},
                "entity_b": {"type": "string", "description": "Second entity or concept."}
            },
            "required": ["entity_a", "entity_b"]
        },
        func=run_compare_entities
    ))

    return registry
