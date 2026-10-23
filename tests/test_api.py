import pytest
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_ui_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Enterprise Agentic RAG" in response.text

def test_ingest_and_query_api():
    # 1. Ingest
    ingest_res = client.post("/api/v1/ingest", json={
        "title": "API Policy",
        "text": "Rate limit is 100 requests per minute. Authentication requires Bearer tokens.",
        "allowed_roles": ["public"]
    })
    assert ingest_res.status_code == 200
    assert ingest_res.json()["status"] == "success"

    # 2. Query
    query_res = client.post("/api/v1/query", json={
        "query": "What is the rate limit for the API?",
        "user_roles": ["public"]
    })
    assert query_res.status_code == 200
    data = query_res.json()
    assert data["verification_passed"] is True, f"Failed with answer: {data.get('answer')}, faith: {data.get('faithfulness_score')}, traces: {data.get('trace_events')}"
    assert "100" in data["answer"]
    assert len(data["citations"]) > 0

    # 3. Graph
    graph_res = client.get("/api/v1/graph")
    assert graph_res.status_code == 200
    assert "node_count" in graph_res.json()

def test_agent_endpoints():
    # 1. Agent Info
    info_res = client.get("/api/v1/agent/info")
    assert info_res.status_code == 200
    info_data = info_res.json()
    assert "provider" in info_data
    assert len(info_data["registered_tools"]) >= 4

    # Ingest document for test
    client.post("/api/v1/ingest", json={
        "title": "API Gateway Spec",
        "text": "The API gateway strictly enforces 100 requests per second rate limit.",
        "allowed_roles": ["public"]
    })

    # 2. Agent Query
    agent_res = client.post("/api/v1/agent/query", json={
        "query": "What is the rate limit for the API gateway?",
        "user_roles": ["public"]
    })
    assert agent_res.status_code == 200
    agent_data = agent_res.json()
    assert "trajectory" in agent_data
    assert "provider" in agent_data
    assert "100" in agent_data["answer"]

    # 3. Agent Stream (SSE)
    with client.stream("POST", "/api/v1/agent/stream", json={
        "query": "What is the rate limit for the API gateway?",
        "user_roles": ["public"]
    }) as stream_res:
        assert stream_res.status_code == 200
        events = [line for line in stream_res.iter_lines() if line]
        assert any("event:" in e for e in events)
        assert any("done" in e or "DONE" in e for e in events)

def test_chunk_inspection_endpoint():
    # Ingest document
    res = client.post("/api/v1/ingest", json={
        "title": "Database Policy",
        "text": "PostgreSQL database port is 5432 and SSL is enabled by default.",
        "allowed_roles": ["public"]
    })
    assert res.status_code == 200

    # Query to get a valid citation tag
    q_res = client.post("/api/v1/query", json={
        "query": "What is the PostgreSQL database port?",
        "user_roles": ["public"]
    })
    assert q_res.status_code == 200
    citations = q_res.json().get("citations", [])
    assert len(citations) > 0

    first_tag = citations[0]
    # Test GET /api/v1/chunk/{tag}
    chunk_res = client.get(f"/api/v1/chunk/{first_tag}")
    assert chunk_res.status_code == 200
    data = chunk_res.json()
    assert data["status"] == "success"
    assert "5432" in data["text"]
    assert "Database Policy" in data["title"]


