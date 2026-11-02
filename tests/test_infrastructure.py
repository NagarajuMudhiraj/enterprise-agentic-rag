"""Unit tests for the hardening layer: metrics, logging, limits, retries, durability."""

import json
import logging
import os

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.agent.llm import provider as provider_module
from src.agent.llm.provider import LLMProvider
from src.api.middleware import RateLimitMiddleware
from src.api.server import _sanitize_filename
from src.common.config import settings
from src.common.logging_config import JsonFormatter, request_id_var
from src.common.metrics import MetricsRegistry
from src.pipeline import EnterpriseRAGPipeline

# ---------------------------------------------------------------- metrics
def test_metrics_render_prometheus_format():
    registry = MetricsRegistry()
    registry.describe("x_total", "number of x")
    registry.incr("x_total", {"route": 'a"b'})
    registry.observe("y_seconds", 0.3, {"route": "/q"})
    registry.gauge("queue_depth", 7)

    text = registry.render()

    assert "# HELP x_total number of x" in text
    assert 'x_total{route="a\\"b"} 1.0' in text
    assert 'y_seconds_bucket{le="+Inf",route="/q"} 1.0' in text
    assert 'y_seconds_count{route="/q"} 1.0' in text
    assert "queue_depth 7" in text


# ---------------------------------------------------------------- logging
def test_json_formatter_carries_request_id_and_extras():
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", None, None)
    record.route = "/api/v1/query"
    token = request_id_var.set("abc123")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)

    assert payload["message"] == "hello"
    assert payload["request_id"] == "abc123"
    assert payload["route"] == "/api/v1/query"


# ------------------------------------------------------------------ limits
def _limited_app(limit: int) -> TestClient:
    app = Starlette(routes=[Route("/ping", lambda request: PlainTextResponse("pong"))])
    app.add_middleware(RateLimitMiddleware, limit_per_minute=limit)
    return TestClient(app)


def test_rate_limit_rejects_after_budget():
    client = _limited_app(limit=2)
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200

    rejected = client.get("/ping")
    assert rejected.status_code == 429
    assert rejected.json()["error"]["code"] == "rate_limited"
    assert rejected.headers["Retry-After"] == "60"


def test_rate_limit_is_per_caller():
    client = _limited_app(limit=1)
    assert client.get("/ping", headers={"X-API-Key": "a"}).status_code == 200
    assert client.get("/ping", headers={"X-API-Key": "b"}).status_code == 200
    assert client.get("/ping", headers={"X-API-Key": "a"}).status_code == 429


def test_rate_limit_exempts_probe_endpoints():
    app = Starlette(routes=[Route("/health", lambda request: PlainTextResponse("ok"))])
    app.add_middleware(RateLimitMiddleware, limit_per_minute=1)
    client = TestClient(app)
    for _ in range(5):
        assert client.get("/health").status_code == 200


# -------------------------------------------------------------- filenames
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\evil.md", "evil.md"),
        ("policy (final) v2.md", "policy _final_ v2.md"),
        ("", "upload"),
        ("....", "upload"),
    ],
)
def test_sanitize_filename(raw, expected):
    name = _sanitize_filename(raw)
    assert name == expected
    assert "/" not in name and "\\" not in name


def test_sanitize_filename_truncates_long_names():
    name = _sanitize_filename("a" * 300 + ".md")
    assert len(name) <= 120
    assert name.endswith(".md")


# ----------------------------------------------------------------- config
def test_api_key_parsing_tolerates_malformed_entries(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "a:admin|b:public,engineer|garbage|:orphan|c:")
    roles = settings.api_key_roles()
    assert roles["a"] == ["admin"]
    assert roles["b"] == ["public", "engineer"]
    assert "garbage" not in roles and ":orphan" not in roles


def test_derived_limits(monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 3)
    monkeypatch.setattr(settings, "ALLOWED_UPLOAD_EXTENSIONS", ".PDF, .md")
    assert settings.max_upload_bytes == 3 * 1024 * 1024
    assert settings.allowed_upload_extensions == [".pdf", ".md"]


# --------------------------------------------------------------- provider
class _FakeResponse:
    def __init__(self, status_code, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    async def post(self, url, headers=None, json=None):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _install_fake_client(monkeypatch, provider, outcomes) -> _FakeClient:
    fake = _FakeClient(outcomes)

    async def _get_client():
        return fake

    async def _no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(provider_module, "get_http_client", _get_client)
    monkeypatch.setattr(provider, "_sleep_before_retry", _no_sleep)
    return fake


@pytest.mark.asyncio
async def test_provider_without_keys_is_local():
    provider = LLMProvider()
    assert provider.provider_type == "local"
    result = await provider.chat_completion([{"role": "user", "content": "hi"}])
    assert result["status"] == "local"


@pytest.mark.asyncio
async def test_provider_retries_rate_limit_then_succeeds(monkeypatch):
    provider = LLMProvider(api_key="k", base_url="http://fake", model="m")
    fake = _install_fake_client(
        monkeypatch,
        provider,
        [
            _FakeResponse(429, headers={"retry-after": "0"}, text="slow down"),
            _FakeResponse(
                200,
                {
                    "choices": [{"message": {"content": "answer"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            ),
        ],
    )

    result = await provider.chat_completion(
        [{"role": "user", "content": "hi"}], max_retries=2
    )

    assert result["status"] == "success"
    assert result["content"] == "answer"
    assert fake.calls == 2, "a 429 must be retried"
    assert provider.usage_summary()["total_tokens"] == 15


@pytest.mark.asyncio
async def test_provider_does_not_retry_client_errors(monkeypatch):
    provider = LLMProvider(api_key="k", base_url="http://fake", model="m")
    fake = _install_fake_client(monkeypatch, provider, [_FakeResponse(400, text="bad request")])

    result = await provider.chat_completion(
        [{"role": "user", "content": "hi"}], max_retries=2
    )

    assert result["status"] == "error"
    assert fake.calls == 1, "a 400 is a client bug, not a transient failure"


@pytest.mark.asyncio
async def test_provider_retries_transport_errors(monkeypatch):
    provider = LLMProvider(api_key="k", base_url="http://fake", model="m")
    fake = _install_fake_client(
        monkeypatch,
        provider,
        [
            httpx.ConnectError("connection refused"),
            _FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]}),
        ],
    )

    result = await provider.chat_completion(
        [{"role": "user", "content": "hi"}], max_retries=1
    )

    assert result["status"] == "success"
    assert fake.calls == 2


# ------------------------------------------------------------- durability
def test_save_state_leaves_no_temp_files(fresh_pipeline):
    fresh_pipeline.ingest_document("# Doc\nSome body text.", title="Doc")
    store = fresh_pipeline.storage_dir

    assert os.path.exists(os.path.join(store, "chunks.json"))
    leftovers = [n for n in os.listdir(store) if n.startswith(".tmp-")]
    assert leftovers == []


def test_load_state_survives_a_corrupt_snapshot(tmp_path):
    store = tmp_path / "corrupt"
    store.mkdir()
    (store / "chunks.json").write_text("{not valid json", encoding="utf-8")

    pipeline = EnterpriseRAGPipeline(storage_dir=str(store))

    assert pipeline._state_loaded is False
    assert pipeline.is_ready()
    assert pipeline.stats()["documents"] == 0


def test_ingest_fails_loudly_when_persistence_fails(fresh_pipeline, monkeypatch):
    def _explode(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr(EnterpriseRAGPipeline, "_atomic_write_json", staticmethod(_explode))

    with pytest.raises(OSError):
        fresh_pipeline.ingest_document("# Doc\nBody.", title="Doc")


def test_delete_document_removes_every_chunk(fresh_pipeline):
    result = fresh_pipeline.ingest_document("# Doc\nBody text here.", title="Doc")
    doc_id = result["doc_id"]
    assert fresh_pipeline.stats()["documents"] == 1

    removed = fresh_pipeline.delete_document(doc_id)

    assert removed > 0
    assert fresh_pipeline.stats()["documents"] == 0
    assert fresh_pipeline.delete_document(doc_id) == 0


def test_concurrent_ingest_keeps_index_consistent(fresh_pipeline):
    import threading

    def _ingest(index: int) -> None:
        fresh_pipeline.ingest_document(f"# Doc {index}\nBody for document {index}.", title=f"Doc {index}")

    threads = [threading.Thread(target=_ingest, args=(i,)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    stats = fresh_pipeline.stats()
    assert stats["documents"] == 5
    assert stats["child_chunks"] >= 5
    assert fresh_pipeline.is_ready()
