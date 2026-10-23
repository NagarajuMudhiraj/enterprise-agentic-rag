import pytest
import asyncio
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import StructureAwareChunker
from src.ingestion.vector_indexer import VectorIndexer
from src.ingestion.keyword_indexer import KeywordIndexer
from src.ingestion.graph_indexer import KnowledgeGraphIndexer
from src.router.adaptive_router import AdaptiveRouter
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.fusion.rrf import ReciprocalRankFusion

SAMPLE_DATA = """# Microservices Architecture

## Payment Gateway
PaymentService connects to StripeAPI for tokenized checkout.
PaymentService depends on VaultService for credentials.

## Notification Engine
NotificationWorker processes messages from RabbitMQ.
"""

@pytest.mark.asyncio
async def test_adaptive_router_and_retrieval():
    doc = DocumentParser.parse_markdown(SAMPLE_DATA, title="Architecture Spec")
    chunker = StructureAwareChunker()
    _, children = chunker.chunk_document(doc)

    chunk_store = {c.chunk_id: c for c in children}
    v_idx = VectorIndexer()
    k_idx = KeywordIndexer()
    g_idx = KnowledgeGraphIndexer()

    v_idx.index_chunks(children)
    k_idx.index_chunks(children)
    g_idx.index_chunks(children)

    # Test Router
    router = AdaptiveRouter()
    plan = router.route_query("What does PaymentService depend on?")
    assert plan["active_paths"]["hybrid"] is True

    # Test Hybrid Retriever
    hybrid = HybridRetriever(v_idx, k_idx)
    dense_cands, sparse_cands = await hybrid.search("PaymentService StripeAPI")
    assert len(sparse_cands) > 0

    # Test Graph Retriever
    graph_retriever = GraphRetriever(g_idx, chunk_store)
    graph_cands = await graph_retriever.search(["PaymentService"])
    assert len(graph_cands) > 0

    # Test RRF Fusion
    fused = ReciprocalRankFusion.fuse([dense_cands, sparse_cands, graph_cands], k=60, top_n=5)
    assert len(fused) > 0
    assert fused[0].score > 0
