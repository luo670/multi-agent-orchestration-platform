"""Application configuration.

All secrets (API keys, database passwords) are loaded from environment variables
via python-dotenv / pydantic-settings. There is NO hardcoded secret anywhere
in the codebase.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings.

    Loads from a ``.env`` file in the project root, then falls back to real
    environment variables. Unknown keys in the .env are ignored.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ---- Database ----
    # SQLite for local dev, MySQL for production.
    # Use the *sync* driver in the URL (pymysql); database.py converts to async.
    DATABASE_URL: str = "sqlite:///./agents.db"

    # ---- DashScope (Qwen) via OpenAI-compatible interface ----
    DASHSCOPE_API_KEY: str = ""
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL: str = "qwen-plus"

    # ---- Redis ----
    REDIS_URL: str = "redis://localhost:6379/0"

    # ---- Celery ----
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ---- App ----
    DEBUG: bool = False
    SECRET_KEY: str = "change-this-secret-key-in-production-please"

    # ---- LLM Retry ----
    MAX_RETRIES: int = 3

    # ---- Reviewer ----
    MAX_REVISIONS: int = 3


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()


# Module-level instance — import this everywhere.
settings = get_settings()