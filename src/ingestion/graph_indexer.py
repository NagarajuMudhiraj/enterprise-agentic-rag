import re
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Any
from src.common.models import Chunk, KnowledgeTriple

class KnowledgeGraphIndexer:
    """
    Constructs and indexes an Entity-Relation Knowledge Graph.
    Extracts named entities, relational predicates, and tracks edge provenance.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.chunk_to_entities: Dict[str, Set[str]] = {}
        self.entity_to_chunks: Dict[str, Set[str]] = {}

    def extract_triples_rule_based(self, chunk: Chunk) -> List[KnowledgeTriple]:
        """
        Extracts entities and relational triples using syntactic relation patterns.
        (In production, this can also call an LLM prompt or an NER pipeline).
        """
        triples: List[KnowledgeTriple] = []
        text = chunk.text
        
        # Regex patterns for common technical / business relationships:
        # e.g. "X is defined as Y", "X depends on Y", "X requires Y", "X connects to Y", "X manages Y"
        patterns = [
            (r'([A-Z][a-zA-Z0-9_\-\s]{2,25})\s+(is\s+defined\s+as|is\s+a|is\s+an)\s+([a-zA-Z0-9_\-\s]{3,35})', "IS_A"),
            (r'([A-Z][a-zA-Z0-9_\-\s]{2,25})\s+(depends\s+on|relies\s+on)\s+([A-Z][a-zA-Z0-9_\-\s]{2,25})', "DEPENDS_ON"),
            (r'([A-Z][a-zA-Z0-9_\-\s]{2,25})\s+(requires|must\s+have)\s+([A-Z][a-zA-Z0-9_\-\s]{2,25})', "REQUIRES"),
            (r'([A-Z][a-zA-Z0-9_\-\s]{2,25})\s+(connects\s+to|communicates\s+with)\s+([A-Z][a-zA-Z0-9_\-\s]{2,25})', "CONNECTS_TO"),
            (r'([A-Z][a-zA-Z0-9_\-\s]{2,25})\s+(manages|stores|processes)\s+([A-Z][a-zA-Z0-9_\-\s]{2,25})', "MANAGES"),
        ]

        for regex, rel_type in patterns:
            for match in re.finditer(regex, text):
                sub = match.group(1).strip()
                obj = match.group(3).strip()
                if sub and obj and len(sub) > 1 and len(obj) > 1:
                    triples.append(
                        KnowledgeTriple(
                            subject=sub,
                            predicate=rel_type,
                            object=obj,
                            chunk_id=chunk.chunk_id,
                            confidence=0.85
                        )
                    )

        # Also extract capitalized named entities
        entities = set(re.findall(r'\b[A-Z][a-zA-Z0-9_]{2,20}\b', text))
        self.chunk_to_entities[chunk.chunk_id] = entities
        for ent in entities:
            if ent not in self.entity_to_chunks:
                self.entity_to_chunks[ent] = set()
            self.entity_to_chunks[ent].add(chunk.chunk_id)

        return triples

    def index_chunks(self, chunks: List[Chunk]):
        for chunk in chunks:
            triples = self.extract_triples_rule_based(chunk)
            for t in triples:
                self.graph.add_node(t.subject, entity_name=t.subject)
                self.graph.add_node(t.object, entity_name=t.object)
                self.graph.add_edge(
                    t.subject,
                    t.object,
                    relation=t.predicate,
                    chunk_id=t.chunk_id,
                    confidence=t.confidence
                )

    def query_subgraph(self, entities: List[str], max_hops: int = 2) -> Dict[str, Any]:
        """
        Retrieves the k-hop neighborhood around target entities.
        Returns connected edges, neighbor entities, and associated chunk IDs.
        """
        matched_chunks = set()
        subgraph_nodes = set()
        paths = []

        for entity in entities:
            # Case-insensitive match on node names
            matched_nodes = [n for n in self.graph.nodes if n.lower() == entity.lower()]
            for node in matched_nodes:
                subgraph_nodes.add(node)
                # Add chunks directly tagging this entity
                matched_chunks.update(self.entity_to_chunks.get(node, []))

                # Traverse k-hops
                try:
                    ego = nx.ego_graph(self.graph, node, radius=max_hops)
                    for u, v, k, data in ego.edges(keys=True, data=True):
                        subgraph_nodes.add(u)
                        subgraph_nodes.add(v)
                        chunk_id = data.get("chunk_id")
                        if chunk_id:
                            matched_chunks.add(chunk_id)
                        paths.append(f"{u} -[{data.get('relation', 'RELATED')}]-> {v}")
                except Exception:
                    pass

        return {
            "entities": list(subgraph_nodes),
            "paths": list(set(paths)),
            "chunk_ids": list(matched_chunks)
        }
