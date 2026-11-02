"""Central runtime configuration.

Every value is environment-overridable (12-factor). No credentials or unsafe
defaults live in code: secrets must come from the environment, and production
startup refuses to run with an unsecured configuration.
"""

from typing import Dict, List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    # ------------------------------------------------------------------ App
    APP_NAME: str = "Enterprise-Agentic-RAG"
    APP_ENV: str = "development"
    APP_VERSION: str = "1.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True

    # --------------------------------------------------------------- Auth
    # Anonymous access is only tolerated outside production; see auth.py.
    AUTH_ENABLED: bool = True
    # Entries are "key:role1,role2" separated by "|", e.g.
    #   rag_live_abc:admin,engineer|rag_public_xyz:public
    API_KEYS: str = ""
    # Roles permitted to ingest or delete documents.
    WRITE_ROLES: str = "admin,engineer"
    # Roles granted to unauthenticated callers when AUTH_ENABLED is false.
    DEV_ANONYMOUS_ROLES: str = "admin,public"

    # -------------------------------------------------------------- Limits
    ALLOWED_ORIGINS: str = ""
    RATE_LIMIT_PER_MINUTE: int = 60
    MAX_UPLOAD_MB: int = 25
    MAX_QUERY_CHARS: int = 4000
    ALLOWED_UPLOAD_EXTENSIONS: str = ".pdf,.docx,.md,.markdown,.txt,.rst"

    # ---------------------------------------------------------- Generation
    GROQ_API_KEY: Optional[str] = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_FAST_MODEL: str = "openai/gpt-oss-20b"

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4o"
    CRITIC_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIM: int = 1536

    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 2
    LLM_MAX_BACKOFF_SECONDS: float = 20.0
    # Optional cost accounting; set to your contracted rates per 1K tokens.
    LLM_INPUT_COST_PER_1K: float = 0.0
    LLM_OUTPUT_COST_PER_1K: float = 0.0

    # ------------------------------------------------------------- Storage
    STORAGE_DIR: str = "data/storage"
    # In-process state cannot be shared between OS processes.
    ALLOW_MULTI_WORKER: bool = False

    # Optional external stores (documented in .env.example). Unset = in-process.
    VECTOR_STORE_URI: Optional[str] = None
    GRAPH_STORE_URI: Optional[str] = None
    GRAPH_USER: Optional[str] = None
    GRAPH_PASSWORD: Optional[str] = None
    OPENSEARCH_URI: Optional[str] = None
    REDIS_URI: Optional[str] = None

    # ---------------------------------------------- Retrieval & Fusion
    RRF_K: int = 60
    TOP_K_RETRIEVAL: int = 15
    TOP_K_RERANK: int = 5
    RERANKER_CONFIDENCE_THRESHOLD: float = 0.35

    # ------------------------------------------- Verification & Feedback
    MAX_VERIFICATION_RETRIES: int = 2
    FAITHFULNESS_THRESHOLD: float = 0.85
    RETRIEVAL_TIMEOUT_MS: int = 400

    # --------------------------------------------------------- Token Budgets
    MAX_CONTEXT_TOKENS: int = 3000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------ Derived
    @field_validator("APP_ENV")
    @classmethod
    def _normalise_env(cls, value: str) -> str:
        return (value or "development").strip().lower()

    @property
    def is_production(self) -> bool:
        return self.APP_ENV in {"production", "prod", "staging"}

    @property
    def allowed_origin_list(self) -> List[str]:
        return _split_csv(self.ALLOWED_ORIGINS)

    @property
    def write_roles(self) -> List[str]:
        return _split_csv(self.WRITE_ROLES)

    @property
    def dev_anonymous_roles(self) -> List[str]:
        return _split_csv(self.DEV_ANONYMOUS_ROLES)

    @property
    def allowed_upload_extensions(self) -> List[str]:
        return [ext.lower() for ext in _split_csv(self.ALLOWED_UPLOAD_EXTENSIONS)]

    @property
    def max_upload_bytes(self) -> int:
        return max(0, self.MAX_UPLOAD_MB) * 1024 * 1024

    def api_key_roles(self) -> Dict[str, List[str]]:
        """Parse API_KEYS into {key: [roles]}.

        Malformed entries are skipped rather than failing startup, so a typo in
        one key cannot lock every caller out.
        """
        mapping: Dict[str, List[str]] = {}
        for entry in self.API_KEYS.split("|"):
            entry = entry.strip()
            if not entry or ":" not in entry:
                continue
            key, _, roles = entry.partition(":")
            key = key.strip()
            if not key:
                continue
            mapping[key] = _split_csv(roles)
        return mapping

    def llm_configured(self) -> bool:
        return bool(self.GROQ_API_KEY or self.OPENAI_API_KEY)


settings = Settings()
