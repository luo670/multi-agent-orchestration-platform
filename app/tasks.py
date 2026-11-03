"""Celery tasks — the async execution layer.

Flow:
    1. FastAPI POST /api/v1/tasks  →  create DB row  →  run_pipeline.delay(task_id, topic)
    2. Celery worker picks up the task:
         a. Update task status → "running"
         b. Build an EventPublisher (Redis Pub/Sub)
         c. Run the LangGraph pipeline (asyncio.run wrapper)
         d. On success: status → "completed", publish "task_done"
         e. On failure:   status → "failed",    publish "task_failed"
    3. FastAPI GET /tasks/{id}/stream subscribes to Redis and forwards events via SSE.

Why asyncio.run() inside the task:
    Celery's prefork worker is sync. Our graph (LangGraph + SQLAlchemy async +
    redis.asyncio) is async. Wrapping the coroutine in asyncio.run() creates a
    fresh event loop per task — clean, isolated, and avoids the well-known
    fragility of mixing gevent/eventlet with asyncio libraries.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.celery_app import celery_app
from app.config import settings
from app.events import EventPublisher
from app.models import Task

logger = logging.getLogger(__name__)

# ---- Sync DB access for the Celery worker ----
# The worker process is sync; it uses a sync session to update task status,
# and runs the async graph pipeline inside asyncio.run().
_sync_engine = create_engine(
    settings.DATABASE_URL,  # raw sync URL (pymysql / sqlite)
    pool_pre_ping=True,
    future=True,
)
_SyncSession = sessionmaker(bind=_sync_engine, class_=Session, expire_on_commit=False)


def _set_task_status(task_id: str, status: str, **fields: Any) -> None:
    """Synchronously update a task row. Called from the sync Celery task body."""
    with _SyncSession() as session:
        stmt = update(Task).where(Task.task_id == task_id).values(status=status, **fields)
        session.execute(stmt)
        session.commit()


async def _execute_pipeline(task_id: str, topic: str) -> dict:
    """Run the agent pipeline inside its own event loop.

    Returns the final graph state. Exceptions propagate to run_pipeline_task.
    """
    # Deferred imports so the worker doesn't initialize async singletons
    # (graph checkpointer, async engine) at module import time.
    from app.graph import close_graph, run_pipeline
    from app.utils import install_all_infra

    # Install metrics interceptor + retry once (idempotent).
    install_all_infra()

    publisher = EventPublisher()
    try:
        final_state = await run_pipeline(task_id=task_id, topic=topic, publisher=publisher)
        return final_state
    finally:
        await publisher.close()
        await close_graph()


async def _publish_event(task_id: str, event_type: str, data: dict) -> None:
    pub = EventPublisher()
    try:
        await pub.publish(task_id, event_type, data)
    finally:
        await pub.close()


@celery_app.task(
    name="app.tasks.run_pipeline_task",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
)
def run_pipeline_task(self, task_id: str, topic: str) -> dict:
    """Celery entrypoint: run the full multi-agent pipeline for one task.

    Args:
        task_id: UUID from FastAPI; also used as LangGraph thread_id.
        topic:   Tech topic to write about.

    Returns:
        A summary dict (final status + revision count). Full state is in DB.
    """
    logger.info("Starting pipeline task_id=%s topic=%r", task_id, topic)
    _set_task_status(task_id, "running")

    try:
        final_state = asyncio.run(_execute_pipeline(task_id, topic))
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Pipeline failed task_id=%s", task_id)
        _set_task_status(task_id, "failed", error_message=tb)
        # Publish terminal failure event so SSE clients disconnect cleanly.
        asyncio.run(_publish_event(task_id, "task_failed", {"error": str(exc)}))
        # Retry transient errors (network, broker hiccups) up to max_retries.
        raise self.retry(exc=exc)

    # Persist the final outputs into the DB row.
    final_article = final_state.get("draft", "")
    revision_count = final_state.get("revision_count", 0)
    _set_task_status(
        task_id,
        "completed",
        final_article=final_article,
        research_notes=final_state.get("research_notes", ""),
        draft=final_state.get("draft", ""),
        feedback=final_state.get("feedback", ""),
        revision_count=revision_count,
    )

    # Publish the terminal "task_done" event so SSE clients stop listening.
    asyncio.run(
        _publish_event(task_id, "task_done", {"chars": len(final_article), "revisions": revision_count})
    )

    logger.info("Pipeline completed task_id=%s revisions=%d", task_id, revision_count)
    return {"task_id": task_id, "status": "completed", "revisions": revision_count}