from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Enterprise-Agentic-RAG"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Generation & LLM Providers (Groq & OpenAI)
    GROQ_API_KEY: Optional[str] = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_FAST_MODEL: str = "openai/gpt-oss-20b"

    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-4o"
    CRITIC_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIM: int = 1536


    # Storage Endpoints
    VECTOR_STORE_URI: str = "http://localhost:6333"
    GRAPH_STORE_URI: str = "bolt://localhost:7687"
    GRAPH_USER: str = "neo4j"
    GRAPH_PASSWORD: str = "password123"
    OPENSEARCH_URI: str = "https://localhost:9200"
    REDIS_URI: str = "redis://localhost:6379"

    # Search & Fusion Parameters
    RRF_K: int = 60
    TOP_K_RETRIEVAL: int = 15
    TOP_K_RERANK: int = 5
    RERANKER_CONFIDENCE_THRESHOLD: float = 0.35

    # Verification & Feedback
    MAX_VERIFICATION_RETRIES: int = 2
    FAITHFULNESS_THRESHOLD: float = 0.85
    RETRIEVAL_TIMEOUT_MS: int = 400

    # Token Budgets
    MAX_CONTEXT_TOKENS: int = 3000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

