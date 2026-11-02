import json
import logging
import os
import asyncio
import tempfile
import threading
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
from src.prompts import PROMPTS_VERSION

from src.agent.tools.registry import ToolRegistry
from src.agent.tools.retrieval_tools import build_retrieval_tools
from src.agent.llm.provider import LLMProvider
from src.agent.events.event_bus import AgentEventBus
from src.agent.core.react_engine import ReActAgentEngine

logger = logging.getLogger(__name__)


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
        self.storage_dir = storage_dir or settings.STORAGE_DIR
        # Guards every index mutation: ingest arrives from the threadpool while a
        # query may be reading the same structures.
        self._lock = threading.RLock()
        self._state_loaded = False
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

        # Agent Core Setup (Groq / OpenAI / Local). Created before the generator so
        # generation and reasoning share one pooled HTTP client and cost ledger.
        self.llm_provider = LLMProvider()
        self.event_bus = AgentEventBus()

        # Generation Layer
        self.context_builder = ContextBuilder()
        self.generator = GroundedGenerator(llm_provider=self.llm_provider)

        # Verification & Critic Layer
        self.critic = AnswerVerificationCritic()
        self.feedback_orchestrator = FeedbackLoopOrchestrator(max_retries=settings.MAX_VERIFICATION_RETRIES)

        self._refresh_agent_tools()

        # Try auto-loading existing index state if present
        if self.storage_dir and os.path.exists(self.storage_dir):
            self._state_loaded = self.load_state(self.storage_dir)

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



    def _remove_documents_titled(self, title: str) -> int:
        """
        Removes previously ingested documents with the same title so re-uploads replace them.
        """
        stale_doc_ids = {
            c.doc_id for c in list(self.parent_store.values()) + list(self.chunk_store.values())
            if c.metadata.get("title") == title
        }
        if not stale_doc_ids:
            return 0

        stale_children = [cid for cid, c in self.chunk_store.items() if c.doc_id in stale_doc_ids]
        for cid in stale_children:
            del self.chunk_store[cid]
        for pid in [pid for pid, p in self.parent_store.items() if p.doc_id in stale_doc_ids]:
            del self.parent_store[pid]

        self.vector_indexer.remove_chunks(stale_children)
        self.keyword_indexer.remove_chunks(stale_children)
        # Graph edges carry chunk provenance; rebuild from the remaining chunks
        self.graph_indexer.__init__()
        self.graph_indexer.index_chunks(list(self.chunk_store.values()))
        return len(stale_doc_ids)

    def _ingest_parsed_doc(self, doc: Document) -> Dict[str, Any]:
        with self._lock:
            self._remove_documents_titled(doc.title)
            parent_chunks, child_chunks = self.chunker.chunk_document(doc)

            for p in parent_chunks:
                self.parent_store[p.chunk_id] = p
            for c in child_chunks:
                self.chunk_store[c.chunk_id] = c

            self.vector_indexer.index_chunks(child_chunks)
            self.keyword_indexer.index_chunks(child_chunks)
            self.graph_indexer.index_chunks(child_chunks)
            self._refresh_agent_tools()

            # Persist before acknowledging. A 200 must mean "durable"; the old
            # `except Exception: pass` here reported success after losing the data.
            if self.storage_dir:
                self.save_state(self.storage_dir)

            total_nodes = self.graph_indexer.graph.number_of_nodes()
            total_edges = self.graph_indexer.graph.number_of_edges()

        logger.info(
            "document_ingested",
            extra={"doc_id": doc.doc_id, "title": doc.title},
        )
        return {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "parent_chunks_count": len(parent_chunks),
            "child_chunks_count": len(child_chunks),
            "total_indexed_nodes": total_nodes,
            "total_indexed_edges": total_edges,
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

    def save_state(self, storage_dir: str = None) -> None:
        """
        Persists chunks and graph edges atomically.

        Each file is written to a temporary sibling and then os.replace()d, so a
        crash mid-write leaves the previous snapshot intact instead of a
        truncated JSON file. Failures propagate: callers must not report success.
        """
        target_dir = storage_dir or self.storage_dir
        if not target_dir:
            raise RuntimeError("No storage directory configured for persistence")
        os.makedirs(target_dir, exist_ok=True)

        with self._lock:
            chunks_data = {
                "parents": {k: v.model_dump() for k, v in self.parent_store.items()},
                "children": {k: v.model_dump() for k, v in self.chunk_store.items()},
            }
            graph_edges = [
                {"u": u, "v": v, **data}
                for u, v, data in self.graph_indexer.graph.edges(data=True)
            ]

        self._atomic_write_json(os.path.join(target_dir, "chunks.json"), chunks_data)
        self._atomic_write_json(os.path.join(target_dir, "graph.json"), graph_edges)

    @staticmethod
    def _atomic_write_json(path: str, payload: Any) -> None:
        directory = os.path.dirname(path) or "."
        descriptor, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def load_state(self, storage_dir: str = "data/storage") -> bool:
        """
        Loads persisted chunks and graph from disk. Returns False when no usable
        snapshot exists; the reason is logged rather than swallowed.
        """
        chunks_path = os.path.join(storage_dir, "chunks.json")
        if not os.path.exists(chunks_path):
            return False

        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Update in place: retrievers and the reranker hold references to these dicts
            with self._lock:
                self.parent_store.clear()
                self.parent_store.update({k: Chunk(**v) for k, v in data.get("parents", {}).items()})
                self.chunk_store.clear()
                self.chunk_store.update({k: Chunk(**v) for k, v in data.get("children", {}).items()})

                child_list = list(self.chunk_store.values())
                self.vector_indexer.index_chunks(child_list)
                self.keyword_indexer.index_chunks(child_list)
                self.graph_indexer.index_chunks(child_list)
                self._refresh_agent_tools()
            logger.info(
                "state_loaded",
                extra={"doc_id": f"{len(self.chunk_store)} chunks"},
            )
            return True

        except Exception:
            logger.exception("Failed to load persisted state from %s", storage_dir)
            return False

    # ----------------------------------------------------------- lifecycle
    def is_ready(self) -> bool:
        """True when this process holds a usable (possibly empty) index."""
        with self._lock:
            return isinstance(self.chunk_store, dict) and self.graph_indexer is not None

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "documents": len(self.list_documents()),
                "parent_chunks": len(self.parent_store),
                "child_chunks": len(self.chunk_store),
                "graph_nodes": self.graph_indexer.graph.number_of_nodes(),
                "graph_edges": self.graph_indexer.graph.number_of_edges(),
                "state_loaded": self._state_loaded,
                "storage_dir": self.storage_dir,
            }

    def list_documents(self) -> List[Dict[str, Any]]:
        """One entry per ingested document, derived from parent chunks."""
        documents: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for chunk in self.parent_store.values():
                entry = documents.get(chunk.doc_id)
                if entry is None:
                    documents[chunk.doc_id] = {
                        "doc_id": chunk.doc_id,
                        "title": chunk.metadata.get("title", "Untitled Document"),
                        "allowed_roles": list(chunk.allowed_roles),
                        "parent_chunks": 1,
                    }
                else:
                    entry["parent_chunks"] += 1
            for chunk in self.chunk_store.values():
                entry = documents.get(chunk.doc_id)
                if entry is not None:
                    entry["child_chunks"] = entry.get("child_chunks", 0) + 1
        return sorted(documents.values(), key=lambda d: d["title"].lower())

    def delete_document(self, doc_id: str) -> int:
        """Removes every chunk belonging to a document. Returns the chunk count removed."""
        with self._lock:
            child_ids = [cid for cid, c in self.chunk_store.items() if c.doc_id == doc_id]
            parent_ids = [pid for pid, p in self.parent_store.items() if p.doc_id == doc_id]
            if not child_ids and not parent_ids:
                return 0

            for cid in child_ids:
                del self.chunk_store[cid]
            for pid in parent_ids:
                del self.parent_store[pid]

            self.vector_indexer.remove_chunks(child_ids)
            self.keyword_indexer.remove_chunks(child_ids)
            # Graph edges carry chunk provenance; rebuild from the remaining chunks
            self.graph_indexer.__init__()
            self.graph_indexer.index_chunks(list(self.chunk_store.values()))
            self._refresh_agent_tools()

            if self.storage_dir:
                self.save_state(self.storage_dir)

        removed = len(child_ids) + len(parent_ids)
        logger.info("document_deleted", extra={"doc_id": doc_id})
        return removed


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
                    trace_events=tracer.spans,
                    unsupported_claims=list(verification.missing_evidence_claims),
                    prompt_version=PROMPTS_VERSION,
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
            trace_events=tracer.spans,
            unsupported_claims=[],
            prompt_version=PROMPTS_VERSION,
        )
