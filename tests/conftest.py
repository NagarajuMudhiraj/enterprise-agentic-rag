"""Hermetic test environment.

Environment variables are set *before* any `src` import because `settings` is
instantiated once at import time. Three guarantees:

  * no live LLM calls — provider keys are blanked, so the deterministic local
    synthesiser is exercised instead of a paid endpoint
  * isolated storage — every suite run indexes into a throwaway directory
  * a known credential — tests assert real auth behaviour instead of bypassing it
"""

import os
import shutil
import tempfile

_SESSION_ROOT = tempfile.mkdtemp(prefix="rag-tests-")

os.environ["APP_ENV"] = "test"
os.environ["STORAGE_DIR"] = os.path.join(_SESSION_ROOT, "store")
os.environ["AUTH_ENABLED"] = "true"
os.environ["API_KEYS"] = "test-admin-key:admin,engineer|test-public-key:public"
os.environ["RATE_LIMIT_PER_MINUTE"] = "100000"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["LOG_JSON"] = "false"
os.environ["LLM_MAX_RETRIES"] = "0"
# `settings` prefers real env vars over .env, so these blank the developer's keys.
os.environ["GROQ_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

import pytest  # noqa: E402  (must follow the env setup above)

ADMIN_KEY = "test-admin-key"
PUBLIC_KEY = "test-public-key"
ADMIN_HEADERS = {"X-API-Key": ADMIN_KEY}
PUBLIC_HEADERS = {"X-API-Key": PUBLIC_KEY}


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_SESSION_ROOT, ignore_errors=True)


@pytest.fixture()
def fresh_pipeline(tmp_path):
    """A pipeline with its own index and storage, isolated from the shared one."""
    from src.pipeline import EnterpriseRAGPipeline

    return EnterpriseRAGPipeline(storage_dir=str(tmp_path / "store"))


@pytest.fixture()
def clean_server_index():
    """Empty the API's module-level pipeline and its snapshot before a test."""
    import os

    from src.api.server import rag_pipeline

    def _reset() -> None:
        rag_pipeline.chunk_store.clear()
        rag_pipeline.parent_store.clear()
        rag_pipeline.vector_indexer = type(rag_pipeline.vector_indexer)()
        rag_pipeline.keyword_indexer = type(rag_pipeline.keyword_indexer)()
        rag_pipeline.graph_indexer = type(rag_pipeline.graph_indexer)()
        rag_pipeline._refresh_agent_tools()
        for name in ("chunks.json", "graph.json"):
            path = os.path.join(rag_pipeline.storage_dir, name)
            if os.path.exists(path):
                os.unlink(path)

    _reset()
    yield rag_pipeline
    _reset()
