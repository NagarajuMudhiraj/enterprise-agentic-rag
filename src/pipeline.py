import os
import asyncio
import time
import uuid

from typing import List, Dict, Optional, Any

from src.common.models import (
    Document,
    Chunk,
    RetrievalCandidate,
    VerifiedResponse,
)
from src.common.config import settings
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import StructureAwareChunker
from src.ingestion.vector_indexer import VectorIndexer
from src.ingestion.keyword_indexer import KeywordIndexer
from src.ingestion.graph_indexer import KnowledgeGraphIndexer

from src.router.adaptive_router import AdaptiveRouter
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.deep_retriever import DeepRetriever

from src.fusion.rrf import ReciprocalRankFusion
from src.fusion.reranker import CrossEncoderReranker
from src.fusion.validator import EvidenceValidator

from src.generation.context_builder import ContextBuilder
from src.generation.generator import GroundedGenerator

from src.verification.critic import AnswerVerificationCritic
from src.verification.feedback_loop import FeedbackLoopOrchestrator

from src.monitoring.tracer import PipelineTracer
from src.monitoring.security import SecurityManager

from src.agent.tools.registry import ToolRegistry
from src.agent.tools.retrieval_tools import build_retrieval_tools
from src.agent.llm.provider import LLMProvider
from src.agent.events.event_bus import AgentEventBus
from src.agent.core.react_engine import ReActAgentEngine

class EnterpriseRAGPipeline:
    """
    Autonomous Enterprise Agentic RAG Pipeline coordinating:
    - Multi-Strategy Ingestion & Tri-Indexing
    - Dynamic Tool Registry & Function Calling (Groq / OpenAI / Local)
    - ReAct Reasoning & Plan-and-Solve Core
    - Corrective RAG (CRAG) & Self-RAG Reflection
    - Event Streaming & Security Guardrails
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir
        # Storage & Ingestion Layer
        self.chunk_store: Dict[str, Chunk] = {}
        self.parent_store: Dict[str, Chunk] = {}
        self.vector_indexer = VectorIndexer()
        self.keyword_indexer = KeywordIndexer()
        self.graph_indexer = KnowledgeGraphIndexer()
        self.chunker = StructureAwareChunker()

        # Routing Layer
        self.router = AdaptiveRouter()

        # Retrieval Layer
        self.hybrid_retriever = HybridRetriever(self.vector_indexer, self.keyword_indexer)
        self.graph_retriever = GraphRetriever(self.graph_indexer, self.chunk_store)
        self.deep_retriever = DeepRetriever(self.parent_store, self.vector_indexer)

        # Fusion & Validation Layer
        self.reranker = CrossEncoderReranker(self.parent_store)

        # Generation Layer
        self.context_builder = ContextBuilder()
        self.generator = GroundedGenerator()

        # Verification & Critic Layer
        self.critic = AnswerVerificationCritic()
        self.feedback_orchestrator = FeedbackLoopOrchestrator(max_retries=settings.MAX_VERIFICATION_RETRIES)

        # Agent Core Setup (Groq / OpenAI / Local)
        self.llm_provider = LLMProvider()
        self.event_bus = AgentEventBus()
        self._refresh_agent_tools()

        # Try auto-loading existing index state if present
        if self.storage_dir and os.path.exists(self.storage_dir):
            self.load_state(self.storage_dir)

    def _refresh_agent_tools(self):
        """
        Rebinds active indices and chunks to the agent tool registry and ReAct engine.
        """
        self.tool_registry = build_retrieval_tools(
            vector_indexer=self.vector_indexer,
            keyword_indexer=self.keyword_indexer,
            graph_indexer=self.graph_indexer,
            chunk_store=self.chunk_store,
            parent_store=self.parent_store
        )
        self.agent = ReActAgentEngine(
            tool_registry=self.tool_registry,
            llm_provider=self.llm_provider,
            event_bus=self.event_bus,
            critic=self.critic
        )



    def _ingest_parsed_doc(self, doc: Document) -> Dict[str, Any]:
        parent_chunks, child_chunks = self.chunker.chunk_document(doc)

        for p in parent_chunks:
            self.parent_store[p.chunk_id] = p
        for c in child_chunks:
            self.chunk_store[c.chunk_id] = c

        self.vector_indexer.index_chunks(child_chunks)
        self.keyword_indexer.index_chunks(child_chunks)
        self.graph_indexer.index_chunks(child_chunks)
        self._refresh_agent_tools()

        # Persist updated state if storage_dir is configured
        if self.storage_dir:
            try:
                self.save_state(self.storage_dir)
            except Exception:
                pass



        return {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "parent_chunks_count": len(parent_chunks),
            "child_chunks_count": len(child_chunks),
            "total_indexed_nodes": len(self.graph_indexer.graph.nodes),
            "total_indexed_edges": len(self.graph_indexer.graph.edges),
        }

    def ingest_document(self, raw_text: str, title: str = "Untitled Document", allowed_roles: List[str] = None) -> Dict[str, Any]:
        """
        Stage 1: Parses, chunks, and simultaneously indexes document into Vector, Keyword, and Graph stores.
        """
        allowed_roles = allowed_roles or ["public"]
        doc = DocumentParser.parse_markdown(raw_text, title=title, allowed_roles=allowed_roles)
        return self._ingest_parsed_doc(doc)

    def ingest_file(self, file_path: str, title: Optional[str] = None, allowed_roles: List[str] = None) -> Dict[str, Any]:
        """
        Ingests a real-world file (.pdf, .md, .txt) into the system in real time.
        """
        doc = DocumentParser.parse_file(file_path, title=title, allowed_roles=allowed_roles)
        return self._ingest_parsed_doc(doc)

    def ingest_directory(self, dir_path: str, allowed_roles: List[str] = None) -> List[Dict[str, Any]]:
        """
        Recursively ingests all documents in a directory.
        """
        docs = DocumentParser.parse_directory(dir_path, allowed_roles=allowed_roles)
        return [self._ingest_parsed_doc(doc) for doc in docs]

    def save_state(self, storage_dir: str = "data/storage") -> None:
        """
        Persists all chunks, vector indices, BM25 indices, and Knowledge Graph to disk.
        """
        import json
        os.makedirs(storage_dir, exist_ok=True)

        # Save chunks
        chunks_data = {
            "parents": {k: v.model_dump() for k, v in self.parent_store.items()},
            "children": {k: v.model_dump() for k, v in self.chunk_store.items()},
        }
        with open(os.path.join(storage_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks_data, f)

        # Save graph edges
        graph_edges = [
            {"u": u, "v": v, **data}
            for u, v, data in self.graph_indexer.graph.edges(data=True)
        ]
        with open(os.path.join(storage_dir, "graph.json"), "w", encoding="utf-8") as f:
            json.dump(graph_edges, f)

    def load_state(self, storage_dir: str = "data/storage") -> bool:
        """
        Loads persisted chunks, vector indices, BM25 indices, and Knowledge Graph from disk.
        """
        import json
        chunks_path = os.path.join(storage_dir, "chunks.json")
        if not os.path.exists(chunks_path):
            return False

        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.parent_store = {k: Chunk(**v) for k, v in data.get("parents", {}).items()}
            self.chunk_store = {k: Chunk(**v) for k, v in data.get("children", {}).items()}

            child_list = list(self.chunk_store.values())
            self.vector_indexer.index_chunks(child_list)
            self.keyword_indexer.index_chunks(child_list)
            self.graph_indexer.index_chunks(child_list)
            self._refresh_agent_tools()
            return True

        except Exception:
            return False


    async def _execute_retrieval_layer(
        self, 
        query: str, 
        routing_plan: Dict[str, Any], 
        user_roles: List[str]
    ) -> List[RetrievalCandidate]:
        """
        Executes active retrieval strategies in parallel with timeouts.
        """
        active_paths = routing_plan["active_paths"]
        tasks = []
        task_names = []

        # 1. Hybrid Search (BM25 + Vectors)
        if active_paths.get("hybrid", True):
            tasks.append(
                self.hybrid_retriever.search(
                    query=query, 
                    top_k=settings.TOP_K_RETRIEVAL, 
                    role_filter=user_roles,
                    hyde_query=routing_plan.get("hyde_passage")
                )
            )
            task_names.append("hybrid")

        # 2. Graph Search (Entities + Paths)
        if active_paths.get("graph", False) and routing_plan.get("entities"):
            tasks.append(
                self.graph_retriever.search(
                    entities=routing_plan["entities"],
                    top_k=settings.TOP_K_RETRIEVAL,
                    role_filter=user_roles
                )
            )
            task_names.append("graph")

        # 3. Deep Retrieval (Hierarchical + Iterative)
        if active_paths.get("deep", False) and routing_plan.get("sub_queries"):
            tasks.append(
                self.deep_retriever.search(
                    sub_queries=routing_plan["sub_queries"],
                    top_k=settings.TOP_K_RETRIEVAL,
                    role_filter=user_roles
                )
            )
            task_names.append("deep")

        # Gather parallel retrieval with timeout budget (400ms)
        timeout_sec = settings.RETRIEVAL_TIMEOUT_MS / 1000.0
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_sec)
        except asyncio.TimeoutError:
            results = []

        candidate_lists: List[List[RetrievalCandidate]] = []
        for name, res in zip(task_names, results):
            if isinstance(res, Exception):
                continue
            if name == "hybrid":
                # res is [dense_cands, sparse_cands]
                candidate_lists.extend(res)
            elif isinstance(res, list):
                candidate_lists.append(res)

        # Stage 3: Reciprocal Rank Fusion (RRF)
        fused_candidates = ReciprocalRankFusion.fuse(candidate_lists, k=settings.RRF_K, top_n=20)
        
        # Deduplication & Provenance Check
        validated = EvidenceValidator.validate_and_deduplicate(fused_candidates, role_filter=user_roles)

        # Neural Cross-Encoder Reranker & Parent Expansion
        reranked = self.reranker.rerank(
            query=query, 
            candidates=validated, 
            top_k=settings.TOP_K_RERANK, 
            threshold=settings.RERANKER_CONFIDENCE_THRESHOLD
        )

        return reranked

    async def query(
        self, 
        user_query: str, 
        user_roles: Optional[List[str]] = None,
        event_bus: Optional[AgentEventBus] = None
    ) -> VerifiedResponse:
        """
        Executes autonomous Agentic RAG query via the ReAct reasoning core.
        """
        user_roles = user_roles or ["public"]
        return await self.agent.run(user_query, user_roles=user_roles, event_bus=event_bus)

    async def agent_query(
        self, 
        user_query: str, 
        user_roles: Optional[List[str]] = None,
        event_bus: Optional[AgentEventBus] = None
    ) -> VerifiedResponse:
        """
        Alias for query() executing autonomous Agentic RAG reasoning.
        """
        return await self.query(user_query=user_query, user_roles=user_roles, event_bus=event_bus)

    async def query_pipeline(self, user_query: str, user_roles: Optional[List[str]] = None) -> VerifiedResponse:
        """
        Executes fixed multi-stage pipeline flow with verification and closed-loop feedback.
        """
        user_roles = user_roles or ["public"]

        query_id = str(uuid.uuid4())[:8]
        tracer = PipelineTracer(query_id=query_id)
        current_query = user_query

        for attempt in range(self.feedback_orchestrator.max_retries + 1):
            iter_start = time.perf_counter()

            # Stage 2: Query Understanding & Adaptive Routing
            t0 = time.perf_counter()
            routing_plan = self.router.route_query(current_query)
            tracer.record_span("query_routing", (time.perf_counter() - t0) * 1000, {"intent": routing_plan["primary_strategy"]})

            # Stage 3: Retrieval, Fusion & Reranking
            t1 = time.perf_counter()
            evidence_candidates = await self._execute_retrieval_layer(current_query, routing_plan, user_roles)
            tracer.record_span("retrieval_fusion_rerank", (time.perf_counter() - t1) * 1000, {"candidates_found": len(evidence_candidates)})

            # Stage 4: Context Assembly & Grounded Generation
            t2 = time.perf_counter()
            formatted_context, citation_manifest = self.context_builder.build_context(evidence_candidates)
            draft_answer, citations = await self.generator.generate_response(user_query, formatted_context, citation_manifest)
            tracer.record_span("context_and_generation", (time.perf_counter() - t2) * 1000, {"citations_generated": citations})

            # Stage 5: Answer Verification Critic
            t3 = time.perf_counter()
            verification = self.critic.verify_answer(draft_answer, evidence_candidates)
            tracer.record_span("verification_critic", (time.perf_counter() - t3) * 1000, {"passed": verification.passed, "faithfulness": verification.faithfulness_score})

            # Decision: Passed
            if verification.passed:
                return VerifiedResponse(
                    query=user_query,
                    answer=SecurityManager.mask_pii(draft_answer),
                    citations=citations,
                    faithfulness_score=verification.faithfulness_score,
                    verification_passed=True,
                    retries_used=attempt,
                    fallback_used=False,
                    latency_ms=tracer.get_total_latency_ms(),
                    trace_events=tracer.spans
                )

            # Decision: Feedback loop retry if allowed
            if self.feedback_orchestrator.should_retry(attempt, verification):
                current_query = self.feedback_orchestrator.reformulate_query(user_query, verification.missing_evidence_claims)
                tracer.record_span("feedback_reformulation", 0.0, {"reformulated_query": current_query})
            else:
                break

        # Max retries exceeded -> Graceful Abstention
        abstain_text = self.feedback_orchestrator.create_abstention_response(user_query, attempt)
        return VerifiedResponse(
            query=user_query,
            answer=abstain_text,
            citations=[],
            faithfulness_score=0.0,
            verification_passed=False,
            retries_used=self.feedback_orchestrator.max_retries,
            fallback_used=True,
            latency_ms=tracer.get_total_latency_ms(),
            trace_events=tracer.spans
        )
