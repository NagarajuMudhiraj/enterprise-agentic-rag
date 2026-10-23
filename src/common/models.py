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
    retries_used: int = 0
    fallback_used: bool = False
    latency_ms: float = 0.0
    trace_events: List[Dict[str, Any]] = Field(default_factory=list)
