"""Database engine, session factory, and declarative base.

Supports both SQLite (local dev) and MySQL (production) by auto-converting
the sync DATABASE_URL from config into its async equivalent.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


def _to_async_url(url: str) -> str:
    """Convert a sync DATABASE_URL to its async driver equivalent.

    Examples:
        mysql+pymysql://...  ->  mysql+aiomysql://...
        sqlite:///./x.db     ->  sqlite+aiosqlite:///./x.db
    """
    if url.startswith("mysql+pymysql"):
        return url.replace("mysql+pymysql", "mysql+aiomysql", 1)
    if url.startswith("mysql://"):
        return url.replace("mysql://", "mysql+aiomysql://", 1)
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
    # Already async or unknown — return as-is.
    return url


# Async engine (used by FastAPI + LangGraph checkpointer).
engine = create_async_engine(
    _to_async_url(settings.DATABASE_URL),
    echo=settings.DEBUG,
    pool_pre_ping=True,
    future=True,
)

# Session factory for dependency injection.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields an async session and auto-closes it."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Create all tables. Call once on startup (dev only).

    In production, prefer Alembic migrations.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_db() -> None:
    """Cleanly dispose of the engine connection pool on shutdown."""
    await engine.dispose()