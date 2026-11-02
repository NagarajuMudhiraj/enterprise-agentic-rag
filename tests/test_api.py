"""API contract tests.

These assert *our* guarantees (status codes, auth, error shape, access control)
rather than the wording of a model's answer, so they stay meaningful when the
prompt or provider changes.
"""

import io

import pytest
from fastapi.testclient import TestClient

from src.api.server import app

ADMIN_HEADERS = {"X-API-Key": "test-admin-key"}
PUBLIC_HEADERS = {"X-API-Key": "test-public-key"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _isolated(clean_server_index):
    """Every test starts from an empty index."""
    return clean_server_index


# ------------------------------------------------------------------ basics
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["version"]


def test_request_id_is_echoed_and_generated(client):
    echoed = client.get("/health", headers={"X-Request-ID": "trace-123"})
    assert echoed.headers["X-Request-ID"] == "trace-123"

    generated = client.get("/health")
    assert generated.headers["X-Request-ID"]

    assert generated.headers["X-Content-Type-Options"] == "nosniff"
    assert "X-Response-Time-Ms" in generated.headers


def test_root_ui_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Enterprise Agentic RAG" in response.text


def test_ready_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


# -------------------------------------------------------------------- auth
def test_query_requires_a_key(client):
    response = client.post("/api/v1/query", json={"query": "anything"})
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "unauthorized"
    assert body["request_id"]


def test_unknown_key_is_rejected(client):
    response = client.post(
        "/api/v1/query",
        json={"query": "anything"},
        headers={"X-API-Key": "not-a-real-key"},
    )
    assert response.status_code == 401


def test_read_only_key_cannot_ingest(client):
    response = client.post(
        "/api/v1/ingest",
        json={"title": "Blocked", "text": "should not be stored"},
        headers=PUBLIC_HEADERS,
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


def test_metrics_requires_admin(client):
    assert client.get("/metrics", headers=PUBLIC_HEADERS).status_code == 403

    response = client.get("/metrics", headers=ADMIN_HEADERS)
    assert response.status_code == 200
    assert "http_requests_total" in response.text


# --------------------------------------------------------------- validation
def test_blank_query_is_a_validation_error(client):
    response = client.post("/api/v1/query", json={"query": "   "}, headers=PUBLIC_HEADERS)
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["fields"]


def test_query_length_is_bounded(client):
    # MAX_QUERY_CHARS is bound into the schema at import time (4000 by default).
    response = client.post("/api/v1/query", json={"query": "x" * 5000}, headers=PUBLIC_HEADERS)
    assert response.status_code == 422


def test_ingest_rejects_blank_title(client):
    response = client.post(
        "/api/v1/ingest",
        json={"title": "  ", "text": "body"},
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 422


# ---------------------------------------------------------------- lifecycle
def test_ingest_query_and_evidence_roundtrip(client):
    ingest = client.post(
        "/api/v1/ingest",
        json={
            "title": "API Policy",
            "text": (
                "# API Policy\n\n"
                "The rate limit is 100 requests per minute.\n"
                "Authentication requires a Bearer token.\n"
            ),
            "allowed_roles": ["public"],
        },
        headers=ADMIN_HEADERS,
    )
    assert ingest.status_code == 200
    assert ingest.json()["status"] == "success"

    query = client.post(
        "/api/v1/query",
        json={"query": "What is the rate limit for the API?"},
        headers=PUBLIC_HEADERS,
    )
    assert query.status_code == 200
    data = query.json()

    # Contract fields the UI depends on.
    for field in ("answer", "citations", "faithfulness_score", "verification_passed",
                  "abstained", "latency_ms", "unsupported_claims", "prompt_version"):
        assert field in data, f"missing response field: {field}"
    assert data["abstained"] is False
    assert data["answer"].strip()
    assert data["citations"], "an answerable question must cite evidence"
    assert data["prompt_version"]

    # The cited evidence must actually contain the fact.
    evidence_texts = []
    for tag in data["citations"]:
        chunk = client.get(f"/api/v1/chunk/{tag}", headers=PUBLIC_HEADERS)
        assert chunk.status_code == 200
        evidence_texts.append(chunk.json()["text"])
    assert any("100" in text for text in evidence_texts)


def test_documents_listing_and_deletion(client):
    ingest = client.post(
        "/api/v1/ingest",
        json={"title": "Retention Policy", "text": "# Retention\nKeep logs for 30 days."},
        headers=ADMIN_HEADERS,
    )
    doc_id = ingest.json()["data"]["doc_id"]

    listing = client.get("/api/v1/documents", headers=PUBLIC_HEADERS)
    assert listing.status_code == 200
    assert any(d["doc_id"] == doc_id for d in listing.json()["documents"])

    deleted = client.delete(f"/api/v1/documents/{doc_id}", headers=ADMIN_HEADERS)
    assert deleted.status_code == 200
    assert deleted.json()["chunks_removed"] > 0

    assert client.delete(f"/api/v1/documents/{doc_id}", headers=ADMIN_HEADERS).status_code == 404


def test_chunk_access_control_hides_restricted_evidence(client, clean_server_index):
    result = clean_server_index.ingest_document(
        raw_text="# Comp Plan\nExecutive bonus pool is capped at 15 percent.",
        title="Compensation",
        allowed_roles=["executive"],
    )
    chunk = next(
        c for c in clean_server_index.chunk_store.values() if c.doc_id == result["doc_id"]
    )
    tag = f"{chunk.doc_id}#{chunk.chunk_id}"

    # Indistinguishable from "does not exist" for a caller without the role.
    assert client.get(f"/api/v1/chunk/{tag}", headers=PUBLIC_HEADERS).status_code == 404
    assert client.get(f"/api/v1/chunk/{tag}", headers=ADMIN_HEADERS).status_code == 200


# ------------------------------------------------------------------ uploads
def test_upload_rejects_unsupported_extension(client):
    response = client.post(
        "/api/v1/ingest/file",
        files={"file": ("malware.exe", io.BytesIO(b"MZ\x00"), "application/octet-stream")},
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_upload_enforces_size_limit(client, monkeypatch):
    from src.common.config import settings

    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)
    payload = b"# Big\n" + b"x" * (1024 * 1024 + 2048)
    response = client.post(
        "/api/v1/ingest/file",
        files={"file": ("big.md", io.BytesIO(payload), "text/markdown")},
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 413


def test_upload_sanitises_traversal_filenames(client):
    response = client.post(
        "/api/v1/ingest/file",
        files={"file": ("../../etc/evil.md", io.BytesIO(b"# Safe\nharmless body\n"), "text/markdown")},
        data={"allowed_roles": "public"},
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "evil.md"


def test_uploaded_file_is_ingested_and_queryable(client):
    body = b"# Backup Policy\n\nBackups run daily at 02:00 UTC and are retained 30 days.\n"
    response = client.post(
        "/api/v1/ingest/file",
        files={"file": ("backup.md", io.BytesIO(body), "text/markdown")},
        data={"allowed_roles": "public"},
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["data"]["child_chunks_count"] >= 1

    listing = client.get("/api/v1/documents", headers=ADMIN_HEADERS)
    assert any(d["title"] == "backup.md" for d in listing.json()["documents"])


# -------------------------------------------------------------------- agent
def test_agent_info_and_query(client):
    info = client.get("/api/v1/agent/info", headers=PUBLIC_HEADERS)
    assert info.status_code == 200
    assert len(info.json()["registered_tools"]) >= 4

    client.post(
        "/api/v1/ingest",
        json={
            "title": "Gateway Spec",
            "text": "# Gateway\n\nThe API gateway enforces a limit of 100 requests per second.",
        },
        headers=ADMIN_HEADERS,
    )

    response = client.post(
        "/api/v1/agent/query",
        json={"query": "What is the rate limit for the API gateway?"},
        headers=PUBLIC_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"].strip()
    assert "trajectory" in data
    assert "provider" in data
    assert "prompt_version" in data
    assert "unsupported_claims" in data


def test_agent_stream_emits_events(client):
    client.post(
        "/api/v1/ingest",
        json={"title": "Stream Doc", "text": "# Stream\nRetention period is 30 days."},
        headers=ADMIN_HEADERS,
    )

    with client.stream(
        "POST",
        "/api/v1/agent/stream",
        json={"query": "What is the retention period?"},
        headers=PUBLIC_HEADERS,
    ) as stream:
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("text/event-stream")
        events = [line for line in stream.iter_lines() if line]
    assert any(line.startswith("event:") for line in events)
    assert any("done" in line for line in events)


def test_graph_overview_requires_auth(client):
    assert client.get("/api/v1/graph").status_code == 401
    response = client.get("/api/v1/graph", headers=PUBLIC_HEADERS)
    assert response.status_code == 200
    assert "node_count" in response.json()


def test_static_assets_are_revalidated(client):
    """Without Cache-Control a browser keeps the previous app.js after a deploy,
    running stale UI code against a newer API."""
    for path in ("/", "/static/app.js", "/static/index.css"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers["cache-control"] == "no-cache", path
