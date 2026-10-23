import os
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from src.pipeline import EnterpriseRAGPipeline
from src.common.models import VerifiedResponse
from src.common.config import settings

app = FastAPI(
    title="Enterprise Agentic RAG Service",
    description="Multi-Strategy Intelligent RAG API with closed-loop verification",
    version="1.0.0"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# UI Directory and Static Mount
UI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui"))
if os.path.exists(UI_DIR):
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

@app.get("/")
def serve_frontend_ui():
    index_file = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Enterprise Agentic RAG API running."}

# Global pipeline instance with persistent storage
rag_pipeline = EnterpriseRAGPipeline(storage_dir="data/storage")


class IngestRequest(BaseModel):
    title: str = Field(..., json_schema_extra={"example": "SLA and Backup Policies"})
    text: str = Field(..., json_schema_extra={"example": "Backup service operates daily at 02:00 UTC. Data retention is 30 days."})
    allowed_roles: List[str] = Field(default_factory=lambda: ["public"])

class QueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What is the backup schedule and retention period?"})
    user_roles: List[str] = Field(default_factory=lambda: ["public"])


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "indexed_chunks": len(rag_pipeline.chunk_store),
        "graph_nodes": len(rag_pipeline.graph_indexer.graph.nodes)
    }

@app.post("/api/v1/ingest", response_model=Dict[str, Any])
def ingest_document(payload: IngestRequest):
    try:
        result = rag_pipeline.ingest_document(
            raw_text=payload.text,
            title=payload.title,
            allowed_roles=payload.allowed_roles
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ingest/file")
async def ingest_uploaded_file(
    file: UploadFile = File(...),
    allowed_roles: str = Form("public")
):
    """
    Ingests real uploaded documents (.pdf, .md, .txt) in real time.
    """
    import os
    import shutil
    import tempfile

    roles = [r.strip() for r in allowed_roles.split(",") if r.strip()]
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = rag_pipeline.ingest_file(
            file_path=file_path,
            title=file.filename,
            allowed_roles=roles
        )
        return {"status": "success", "filename": file.filename, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest file: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/v1/query", response_model=VerifiedResponse)
async def query_pipeline(payload: QueryRequest):
    try:
        response = await rag_pipeline.query(
            user_query=payload.query,
            user_roles=payload.user_roles
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/graph")
def get_graph_overview():
    nodes = list(rag_pipeline.graph_indexer.graph.nodes(data=True))
    edges = list(rag_pipeline.graph_indexer.graph.edges(data=True))
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [{"id": n[0], **n[1]} for n in nodes[:50]],
        "edges": [{"from": e[0], "to": e[1], **e[2]} for e in edges[:50]]
    }

@app.get("/api/v1/chunk/{tag_or_id}")
def get_chunk_details(tag_or_id: str):
    """
    Retrieves full chunk text, metadata, and parent context for interactive citation inspection.
    """
    clean_id = tag_or_id.strip("[]【】 \t\r\n")
    cid = clean_id.split("#")[-1]

    chunk = rag_pipeline.chunk_store.get(cid) or rag_pipeline.parent_store.get(cid)
    if not chunk:
        for k, c in rag_pipeline.chunk_store.items():
            if cid in k or f"{c.doc_id}#{c.chunk_id}" == clean_id:
                chunk = c
                break

    if not chunk:
        raise HTTPException(status_code=404, detail=f"Evidence chunk '{clean_id}' not found.")

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
        "metadata": chunk.metadata
    }

@app.get("/api/v1/agent/info")
def get_agent_info():
    """
    Returns active Agent configuration, tools, and LLM provider details.
    """
    tools = rag_pipeline.agent.tools.list_tools()
    return {
        "provider": rag_pipeline.agent.llm.provider_type,
        "model": rag_pipeline.agent.llm.model,
        "max_iterations": rag_pipeline.agent.max_iterations,
        "registered_tools": [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            }
            for t in tools
        ],
        "indexed_chunks": len(rag_pipeline.chunk_store),
        "knowledge_graph_entities": len(rag_pipeline.graph_indexer.graph.nodes)
    }

@app.post("/api/v1/agent/query")
async def agent_query(payload: QueryRequest):
    """
    Autonomous ReAct Agent reasoning endpoint returning complete plan, trajectory,
    graded observations, and verified answer.
    """
    try:
        response = await rag_pipeline.query(
            user_query=payload.query,
            user_roles=payload.user_roles
        )
        return {
            "query": response.query,
            "answer": response.answer,
            "citations": response.citations,
            "faithfulness_score": response.faithfulness_score,
            "verification_passed": response.verification_passed,
            "retries_used": response.retries_used,
            "latency_ms": response.latency_ms,
            "trajectory": response.trace_events,
            "provider": rag_pipeline.agent.llm.provider_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/agent/stream")
async def stream_agent_query(payload: QueryRequest):
    """
    Real-time Server-Sent Events (SSE) endpoint streaming thoughts, tool actions,
    observations, and final verified answer.
    """
    import asyncio
    import json
    from starlette.responses import StreamingResponse
    from src.agent.events.event_bus import AgentEventBus, AgentEvent

    queue = asyncio.Queue()
    scoped_bus = AgentEventBus()

    async def on_event(event: AgentEvent):
        await queue.put(event)

    scoped_bus.subscribe(on_event)

    async def event_generator():
        task = asyncio.create_task(
            rag_pipeline.query(
                user_query=payload.query,
                user_roles=payload.user_roles,
                event_bus=scoped_bus
            )
        )

        while True:
            if task.done() and queue.empty():
                exc = task.exception()
                if exc:
                    yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                break

            try:
                event: AgentEvent = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield f"event: {event.event_type.lower()}\ndata: {json.dumps(event.data)}\n\n"
            except asyncio.TimeoutError:
                continue

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
