from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field
import time
import uuid

class Document(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    allowed_roles: List[str] = Field(default_factory=lambda: ["public"])

class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    doc_id: str
    parent_id: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    allowed_roles: List[str] = Field(default_factory=lambda: ["public"])
    embedding: Optional[List[float]] = None

class KnowledgeTriple(BaseModel):
    subject: str
    predicate: str
    object: str
    chunk_id: str
    confidence: float = 1.0

class QueryIntent(BaseModel):
    raw_query: str
    intent_type: Literal["factoid", "relational", "synthesized_summary"]
    sub_queries: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    needs_hybrid: bool = True
    needs_graph: bool = False
    needs_deep: bool = False
    confidence: float = 1.0

class RetrievalCandidate(BaseModel):
    chunk: Chunk
    source_strategy: str  # 'hybrid', 'graph', 'deep'
    score: float = 0.0
    rank: int = 0
    parent_text: Optional[str] = None

class AtomicClaim(BaseModel):
    claim_text: str
    cited_chunk_ids: List[str] = Field(default_factory=list)
    entailment_status: Literal["entailed", "neutral", "contradicted"] = "neutral"
    entailment_score: float = 0.0
    supporting_quote: Optional[str] = None

class VerificationResult(BaseModel):
    faithfulness_score: float
    claims: List[AtomicClaim] = Field(default_factory=list)
    missing_evidence_claims: List[str] = Field(default_factory=list)
    passed: bool
    verdict_explanation: str

class VerifiedResponse(BaseModel):
    query: str
    answer: str
    citations: List[str] = Field(default_factory=list)
    faithfulness_score: float
    verification_passed: bool
    # True when the system declined to answer because no retrieved passage
    # supported the question. That is a different outcome from "answered, but
    # ungrounded": an abstention makes no factual claims, so there is nothing
    # for the critic to grade and a faithfulness number would be misleading.
    abstained: bool = False
    retries_used: int = 0
    fallback_used: bool = False
    latency_ms: float = 0.0
    trace_events: List[Dict[str, Any]] = Field(default_factory=list)
    # Claims the critic could not tie to retrieved evidence; surfaced to the UI
    # so users can see exactly which sentences are unsupported.
    unsupported_claims: List[str] = Field(default_factory=list)
    prompt_version: Optional[str] = None
