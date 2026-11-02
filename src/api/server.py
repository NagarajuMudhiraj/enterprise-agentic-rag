"""HTTP surface for the Enterprise Agentic RAG service.

Design notes:
  * Authorization is derived from the API key only. The request body never
    carries roles, so a caller cannot grant itself access to a document.
  * Every mutating route is guarded by a write role, operator routes by admin.
  * Errors share one JSON shape so clients can branch on `error.code`.
  * State is in-process; run exactly one worker per container unless you
    replace the index with a shared store (guarded at startup).
"""

import asyncio
import contextlib
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.agent.events.event_bus import AgentEvent, AgentEventBus
from src.agent.llm.provider import close_http_client
from src.api import audit
from src.api.auth import Principal, get_principal, require_admin, require_write
from src.api.middleware import RateLimitMiddleware, RequestContextMiddleware
from src.common.config import settings
from src.common.logging_config import configure_logging, get_logger, request_id_var
from src.common.metrics import metrics
from src.common.models import VerifiedResponse
from src.pipeline import EnterpriseRAGPipeline
from src.prompts import PROMPTS_VERSION

logger = get_logger("api.server")

UI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui"))
_STARTED_AT = time.time()
_UPLOAD_CHUNK_BYTES = 64 * 1024
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._\- ]+")


# --------------------------------------------------------------- startup
def validate_runtime_config() -> None:
    """Fail closed on configurations that are unsafe to serve.

    Called once from the lifespan so a misconfigured container crashes on
    boot rather than silently serving unauthenticated traffic.
    """
    problems: List[str] = []

    workers = int(os.getenv("WEB_CONCURRENCY", "1") or "1")
    if workers > 1 and not settings.ALLOW_MULTI_WORKER:
        problems.append(
            f"WEB_CONCURRENCY={workers} but the index is per-process; set "
            "ALLOW_MULTI_WORKER=true only after moving to a shared store"
        )

    if settings.is_production:
        if not settings.AUTH_ENABLED:
            problems.append("AUTH_ENABLED must be true in production")
        if not settings.api_key_roles():
            problems.append("API_KEYS must define at least one key in production")
        if not settings.allowed_origin_list:
            logger.warning(
                "ALLOWED_ORIGINS is empty: cross-origin browser clients will be rejected"
            )

    if not settings.llm_configured():
        logger.warning(
            "No LLM credentials configured; generation falls back to the "
            "deterministic local synthesiser"
        )

    if problems:
        raise RuntimeError("Unsafe runtime configuration: " + "; ".join(problems))


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    validate_runtime_config()
    logger.info(
        "service_starting",
        extra={
            "event": "startup",
            "provider": getattr(rag_pipeline.llm_provider, "provider_type", "unknown"),
            "model": getattr(rag_pipeline.llm_provider, "model", "unknown"),
            "route": settings.APP_ENV,
        },
    )
    try:
        yield
    finally:
        await close_http_client()
        logger.info("service_stopped", extra={"event": "shutdown"})


configure_logging()

app = FastAPI(
    title="Enterprise Agentic RAG Service",
    description="Multi-strategy agentic RAG API with closed-loop verification.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None,
)

# Middleware is added inner-to-outer: CORS must be outermost to decorate
# error responses produced by the layers beneath it.
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origin_list or ([] if settings.is_production else ["*"]),
    allow_credentials=bool(settings.allowed_origin_list),
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID", "Accept"],
    expose_headers=["X-Request-ID", "X-Response-Time-Ms", "Retry-After"],
    max_age=600,
)

class RevalidatingStaticFiles(StaticFiles):
    """Static assets must be revalidated on every load.

    Starlette sends an ETag but no Cache-Control, so browsers cache /static/*
    heuristically. After a deploy that left returning users running the previous
    app.js against a newer API. `no-cache` still allows a 304, so revalidation
    costs one conditional request, not a re-download.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


if os.path.exists(UI_DIR):
    app.mount("/static", RevalidatingStaticFiles(directory=UI_DIR), name="static")

# Module-level for CLI and test compatibility. Index state is loaded here so a
# warm start answers queries immediately.
rag_pipeline = EnterpriseRAGPipeline(storage_dir=settings.STORAGE_DIR)


# ---------------------------------------------------------- error contract
def _error(
    status_code: int,
    code: str,
    message: str,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"code": code, "message": message},
            "request_id": request_id_var.get(),
        },
        headers=headers,
    )


@app.exception_handler(StarletteHTTPException)
async def _http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    message = detail if isinstance(detail, str) else json.dumps(detail)
    code = {401: "unauthorized", 403: "forbidden", 404: "not_found", 413: "payload_too_large",
            415: "unsupported_media_type", 429: "rate_limited",
            503: "service_unavailable"}.get(exc.status_code, "http_error")
    return _error(exc.status_code, code, message, headers=dict(exc.headers or {}))


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request body failed validation.",
                "fields": [
                    {
                        "location": ".".join(str(p) for p in err.get("loc", [])),
                        "message": err.get("msg", "invalid"),
                    }
                    for err in exc.errors()
                ],
            },
            "request_id": request_id_var.get(),
        },
    )


@app.exception_handler(Exception)
async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error", extra={"route": request.url.path})
    return _error(500, "internal_error", "Unexpected server error.")


# ----------------------------------------------------------- request models
class IngestRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1)
    allowed_roles: List[str] = Field(default_factory=lambda: ["public"])

    @field_validator("title", "text")
    @classmethod
    def _strip(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("text")
    @classmethod
    def _size(cls, value: str) -> str:
        if len(value) > settings.max_upload_bytes:
            raise ValueError("document exceeds MAX_UPLOAD_MB")
        return value

    @field_validator("allowed_roles")
    @classmethod
    def _roles(cls, value: List[str]) -> List[str]:
        roles = [r.strip() for r in value if r and r.strip()]
        return roles or ["public"]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=settings.MAX_QUERY_CHARS)

    @field_validator("query")
    @classmethod
    def _strip(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


# -------------------------------------------------------------- UI assets
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg_icon = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
        "stroke='%23970747' stroke-width='2.2'><polygon points='12 2 2 7 12 12 22 7 12 2'/>"
        "<polyline points='2 17 12 22 22 17'/><polyline points='2 12 12 17 22 12'/></svg>"
    )
    return Response(content=svg_icon, media_type="image/svg+xml")


@app.get("/", include_in_schema=False)
def serve_frontend_ui():
    index_file = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file, headers={"Cache-Control": "no-cache"})
    return {"message": "Enterprise Agentic RAG API running.", "docs": "/docs"}


# ---------------------------------------------------------------- probes
@app.get("/health", tags=["ops"])
def health_check():
    """Liveness: the process is up. Never touches the index or the LLM."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "uptime_seconds": round(time.time() - _STARTED_AT, 1),
    }


@app.get("/ready", tags=["ops"])
def readiness_check():
    """Readiness: the index is usable and can accept traffic."""
    if not rag_pipeline.is_ready():
        return _error(503, "not_ready", "Index is not initialised.")
    stats = rag_pipeline.stats()
    return {
        "status": "ready",
        "documents": stats["documents"],
        "child_chunks": stats["child_chunks"],
        "graph_nodes": stats["graph_nodes"],
        "state_loaded": stats["state_loaded"],
    }


@app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
def prometheus_metrics(_: Principal = Depends(require_admin)):
    metrics.gauge("rag_chunks_indexed", float(len(rag_pipeline.chunk_store)))
    metrics.gauge("rag_documents_total", float(len(rag_pipeline.list_documents())))
    return PlainTextResponse(
        metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# -------------------------------------------------------------- ingestion
@app.post("/api/v1/ingest", response_model=Dict[str, Any], tags=["ingest"])
def ingest_document(
    payload: IngestRequest,
    principal: Principal = Depends(require_write),
):
    try:
        result = rag_pipeline.ingest_document(
            raw_text=payload.text,
            title=payload.title,
            allowed_roles=payload.allowed_roles,
        )
    except Exception:
        logger.exception("ingest_failed", extra={"principal": principal.label})
        metrics.incr("rag_documents_total", {"result": "error"})
        raise HTTPException(500, "Document ingestion failed.")

    metrics.incr("rag_documents_total", {"result": "success"})
    audit.record("ingest", principal.label, doc_id=result["doc_id"], title=result["title"])
    return {"status": "success", "data": result}


def _sanitize_filename(raw: str) -> str:
    """Reduce any client-supplied name to a bare, safe basename."""
    name = os.path.basename((raw or "").replace("\\", "/"))
    name = _UNSAFE_NAME_CHARS.sub("_", name).strip(". ")
    if len(name) > 120:
        stem, dot, ext = name.rpartition(".")
        name = f"{stem[:100]}{dot}{ext[:16]}" if dot else name[:120]
    return name or "upload"


async def _spool_upload(file: UploadFile, destination: str) -> int:
    """Stream an upload to disk, aborting as soon as the size cap is crossed."""
    limit = settings.max_upload_bytes
    written = 0
    with open(destination, "wb") as handle:
        while True:
            chunk = await file.read(_UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                raise HTTPException(
                    413,
                    f"Upload exceeds the {settings.MAX_UPLOAD_MB} MB limit.",
                )
            handle.write(chunk)
    return written


@app.post("/api/v1/ingest/file", response_model=Dict[str, Any], tags=["ingest"])
async def ingest_uploaded_file(
    file: UploadFile = File(...),
    allowed_roles: str = "",
    principal: Principal = Depends(require_write),
):
    """Ingest a .pdf/.docx/.md/.txt upload."""
    filename = _sanitize_filename(file.filename or "")
    extension = os.path.splitext(filename)[1].lower()
    if extension not in settings.allowed_upload_extensions:
        raise HTTPException(
            415,
            f"Unsupported file type '{extension or 'none'}'. "
            f"Allowed: {', '.join(settings.allowed_upload_extensions)}",
        )

    roles = [r.strip() for r in allowed_roles.split(",") if r.strip()] or ["public"]
    temp_dir = os.path.join(settings.STORAGE_DIR or ".", ".uploads")
    os.makedirs(temp_dir, exist_ok=True)
    destination = os.path.join(temp_dir, f"{uuid.uuid4().hex}_{filename}")

    try:
        size = await _spool_upload(file, destination)
        if size == 0:
            raise HTTPException(400, "Uploaded file is empty.")

        result = await run_in_threadpool(
            rag_pipeline.ingest_file,
            file_path=destination,
            title=filename,
            allowed_roles=roles,
        )
    except HTTPException:
        metrics.incr("rag_documents_total", {"result": "rejected"})
        raise
    except Exception:
        logger.exception("file_ingest_failed", extra={"principal": principal.label})
        metrics.incr("rag_documents_total", {"result": "error"})
        raise HTTPException(500, "Document ingestion failed.")
    finally:
        with contextlib.suppress(OSError):
            os.unlink(destination)

    metrics.incr("rag_documents_total", {"result": "success"})
    audit.record("ingest_file", principal.label, doc_id=result["doc_id"], title=filename)
    return {"status": "success", "filename": filename, "data": result}


@app.get("/api/v1/documents", tags=["ingest"])
def list_documents(principal: Principal = Depends(get_principal)):
    documents = rag_pipeline.list_documents()
    if not principal.is_admin:
        documents = [
            d for d in documents
            if set(d["allowed_roles"]) & set(principal.roles)
        ]
    return {"status": "success", "count": len(documents), "documents": documents}


@app.delete("/api/v1/documents/{doc_id}", tags=["ingest"])
def delete_document(doc_id: str, principal: Principal = Depends(require_write)):
    removed = rag_pipeline.delete_document(doc_id)
    if removed == 0:
        raise HTTPException(404, f"Document '{doc_id}' was not found.")
    audit.record("delete_document", principal.label, doc_id=doc_id, chunks_removed=removed)
    return {"status": "success", "doc_id": doc_id, "chunks_removed": removed}


# ------------------------------------------------------------------ query
@app.post("/api/v1/query", response_model=VerifiedResponse, tags=["query"])
async def query_pipeline(
    payload: QueryRequest,
    principal: Principal = Depends(get_principal),
):
    try:
        response = await rag_pipeline.query(
            user_query=payload.query,
            user_roles=list(principal.roles),
        )
    except Exception:
        logger.exception("query_failed", extra={"principal": principal.label})
        raise HTTPException(500, "Query failed. Check server logs for the request id.")

    audit.record(
        "query",
        principal.label,
        query_chars=len(payload.query),
        verification_passed=response.verification_passed,
        faithfulness=round(response.faithfulness_score, 3),
    )
    return response


@app.get("/api/v1/graph", tags=["query"])
def get_graph_overview(principal: Principal = Depends(get_principal)):
    graph = rag_pipeline.graph_indexer.graph
    nodes = list(graph.nodes(data=True))
    edges = list(graph.edges(data=True))
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [{"id": n[0], **n[1]} for n in nodes[:50]],
        "edges": [{"from": e[0], "to": e[1], **e[2]} for e in edges[:50]],
    }


@app.get("/api/v1/chunk/{tag_or_id}", tags=["query"])
def get_chunk_details(tag_or_id: str, principal: Principal = Depends(get_principal)):
    """Full text and parent context for a citation tag [doc#chunk]."""
    clean_id = tag_or_id.strip("[]【】 \t\r\n")
    chunk_id = clean_id.split("#")[-1]

    chunk = rag_pipeline.chunk_store.get(chunk_id) or rag_pipeline.parent_store.get(chunk_id)
    if not chunk:
        for key, candidate in rag_pipeline.chunk_store.items():
            if chunk_id in key or f"{candidate.doc_id}#{candidate.chunk_id}" == clean_id:
                chunk = candidate
                break

    # Unknown and forbidden look identical: chunk existence is not a public fact.
    if not chunk or not (principal.is_admin or set(chunk.allowed_roles) & set(principal.roles)):
        raise HTTPException(404, f"Evidence chunk '{clean_id}' not found.")

    parent = rag_pipeline.parent_store.get(chunk.parent_id) if chunk.parent_id else None
    return {
        "status": "success",
        "citation_tag": f"[{chunk.doc_id}#{chunk.chunk_id}]",
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.metadata.get("title", "Document"),
        "text": chunk.text,
        "parent_id": chunk.parent_id,
        "parent_text": parent.text if parent else None,
        "allowed_roles": chunk.allowed_roles,
        "metadata": chunk.metadata,
    }


# ------------------------------------------------------------------ agent
@app.get("/api/v1/agent/info", tags=["agent"])
def get_agent_info(principal: Principal = Depends(get_principal)):
    tools = rag_pipeline.agent.tools.list_tools()
    return {
        "provider": rag_pipeline.agent.llm.provider_type,
        "model": rag_pipeline.agent.llm.model,
        "max_iterations": rag_pipeline.agent.max_iterations,
        "registered_tools": [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in tools
        ],
        "indexed_chunks": len(rag_pipeline.chunk_store),
        "knowledge_graph_entities": rag_pipeline.graph_indexer.graph.number_of_nodes(),
    }


@app.post("/api/v1/agent/query", tags=["agent"])
async def agent_query(
    payload: QueryRequest,
    principal: Principal = Depends(get_principal),
):
    """Full ReAct trajectory plus the verified answer."""
    try:
        response = await rag_pipeline.query(
            user_query=payload.query,
            user_roles=list(principal.roles),
        )
    except Exception:
        logger.exception("agent_query_failed", extra={"principal": principal.label})
        raise HTTPException(500, "Agent query failed. Check server logs for the request id.")

    audit.record(
        "agent_query",
        principal.label,
        query_chars=len(payload.query),
        verification_passed=response.verification_passed,
    )
    return {
        "query": response.query,
        "answer": response.answer,
        "citations": response.citations,
        "faithfulness_score": response.faithfulness_score,
        "verification_passed": response.verification_passed,
        "abstained": response.abstained,
        "retries_used": response.retries_used,
        "latency_ms": response.latency_ms,
        "trajectory": response.trace_events,
        "unsupported_claims": response.unsupported_claims,
        "prompt_version": response.prompt_version,
        "provider": rag_pipeline.agent.llm.provider_type,
    }


@app.post("/api/v1/agent/stream", tags=["agent"])
async def stream_agent_query(
    payload: QueryRequest,
    request: Request,
    principal: Principal = Depends(get_principal),
):
    """Server-Sent Events stream of plan, thoughts, actions and the final answer."""
    queue: "asyncio.Queue[AgentEvent]" = asyncio.Queue()
    scoped_bus = AgentEventBus()

    async def on_event(event: AgentEvent):
        await queue.put(event)

    scoped_bus.subscribe(on_event)

    async def event_generator():
        task = asyncio.create_task(
            rag_pipeline.query(
                user_query=payload.query,
                user_roles=list(principal.roles),
                event_bus=scoped_bus,
            )
        )
        try:
            while True:
                if await request.is_disconnected():
                    break

                if task.done() and queue.empty():
                    exception = task.exception()
                    if exception is not None:
                        logger.exception("agent_stream_failed", exc_info=exception)
                        yield (
                            "event: error\ndata: "
                            + json.dumps({"error": "Agent query failed."})
                            + "\n\n"
                        )
                    yield "event: done\ndata: [DONE]\n\n"
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                yield f"event: {event.event_type.lower()}\ndata: {json.dumps(event.data, default=str)}\n\n"
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            scoped_bus.unsubscribe(on_event)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ------------------------------------------------------------------ admin
@app.get("/api/v1/admin/usage", tags=["admin"])
def get_usage(_: Principal = Depends(require_admin)):
    return {
        "status": "success",
        "prompt_version": PROMPTS_VERSION,
        "llm": rag_pipeline.llm_provider.usage_summary(),
        "index": rag_pipeline.stats(),
    }


@app.get("/api/v1/admin/audit", tags=["admin"])
def get_audit_log(limit: int = 100, _: Principal = Depends(require_admin)):
    limit = max(1, min(limit, 500))
    return {"status": "success", "events": audit.recent(limit)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=not settings.is_production,
        workers=1,
    )
