"""FastAPI application — the HTTP gateway.

Two endpoints (per spec):
    POST /api/v1/tasks                   — submit a topic, return task_id
    GET  /api/v1/tasks/{task_id}/stream  — SSE stream of agent progress

Plus small helpers:
    GET  /api/v1/tasks/{task_id}         — fetch final result (optional)
    GET  /health                         — liveness probe

SSE design (Redis Pub/Sub bridge):
    The Celery worker publishes events to Redis channel "task:{id}:events".
    The SSE endpoint subscribes to that channel and forwards each message
    as an SSE ``data:`` frame. Why not poll the DB? Because:
      - Pub/Sub is push-based → sub-second latency to the browser
      - Decouples producer (worker) from consumer (FastAPI) — multiple
        browser tabs can watch the same task without extra DB load
      - The worker never blocks on streaming
    A lightweight DB fallback poll is layered on top so that if a client
    connects AFTER the worker already finished (events already published and
    gone), the endpoint still returns the final state and closes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from app.database import AsyncSessionLocal, dispose_db, init_db
from app.events import EventSubscriber
from app.models import Task
from app.tasks import run_pipeline_task

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ============================================================
# Request / response schemas
# ============================================================

class TaskCreateRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=512, description="Tech topic to write about")


class TaskCreateResponse(BaseModel):
    task_id: str
    status: str
    topic: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    topic: str
    revision_count: int
    final_article: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


# ============================================================
# App lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables (dev only). Production uses Alembic.
    logger.info("Initializing database tables...")
    await init_db()
    yield
    # Shutdown: release resources.
    await dispose_db()
    logger.info("Database disposed. Bye.")


app = FastAPI(
    title="Multi-Agent Orchestration Platform",
    description="Enterprise multi-agent collaboration system for tech media editorial teams",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the frontend (and tools) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Helpers
# ============================================================

async def _get_task(task_id: str) -> Task | None:
    async with AsyncSessionLocal() as session:
        return await session.scalar(select(Task).where(Task.task_id == task_id))


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.post("/api/v1/tasks", response_model=TaskCreateResponse, status_code=201)
async def create_task(req: TaskCreateRequest):
    """Submit a topic and enqueue the pipeline.

    1. Insert a DB row (status=pending)
    2. Dispatch to Celery via run_pipeline_task.delay(...)
    3. Return the task_id immediately (client then connects to the SSE stream)
    """
    async with AsyncSessionLocal() as session:
        task = Task(topic=req.topic)
        session.add(task)
        await session.commit()
        await session.refresh(task)
        task_id = task.task_id

    # Hand off to Celery — non-blocking.
    run_pipeline_task.delay(task_id, req.topic)

    return TaskCreateResponse(task_id=task_id, status="pending", topic=req.topic)


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str):
    """Fetch the final status / result of a task."""
    task = await _get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        topic=task.topic,
        revision_count=task.revision_count,
        final_article=task.final_article,
        error_message=task.error_message,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


@app.get("/api/v1/tasks/{task_id}/stream")
async def stream_task(task_id: str):
    """SSE endpoint: stream real-time agent progress to the client.

    Strategy (Redis Pub/Sub + DB fallback):
      1. Subscribe to the task's Redis channel.
      2. Forward each Pub/Sub message as an SSE ``data:`` frame.
      3. If the task is already terminal (completed/failed) when the client
         connects — i.e., events were published before subscription — emit
         a synthetic terminal event and close.
      4. The EventSubscriber auto-stops after a terminal event (task_done /
         task_failed), so the SSE stream closes cleanly.

    SSE frame format:
        data: {"type": "node_start", "data": {...}, "ts": 1730000000.0}\\n\\n

    A keep-alive comment (``: ping``) is sent every 15s to prevent proxy
    timeouts (e.g. nginx default 60s).
    """
    # Verify the task exists first.
    task = await _get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    # If already done, send the terminal frame immediately and close.
    if task.status in ("completed", "failed"):
        async def _immediate():
            evt_type = "task_done" if task.status == "completed" else "task_failed"
            data = {"final_article_chars": len(task.final_article or "")} if task.status == "completed" else {"error": task.error_message}
            yield {
                "event": evt_type,
                "data": json.dumps({"type": evt_type, "data": data, "ts": datetime.now(timezone.utc).timestamp()}),
            }
        return EventSourceResponse(_immediate())

    # Otherwise, bridge Redis Pub/Sub → SSE.
    subscriber = EventSubscriber()

    async def event_generator():
        try:
            last_ping = asyncio.get_event_loop().time()
            async for raw in subscriber.listen(task_id):
                # Parse to detect terminal events (subscriber already breaks,
                # but we also check defensively).
                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    evt = {"type": "unknown", "data": raw}

                yield {"data": raw}

                # Keep-alive: SSE comment line (done in the loop below).

                # Break on terminal (EventSubscriber already does this, but
                # double-guard).
                if evt.get("type") in ("task_done", "task_failed"):
                    break

                # Slow clients: send a ping every 15s if no events arrived.
                now = asyncio.get_event_loop().time()
                if now - last_ping > 15:
                    yield {"comment": "ping"}
                    last_ping = now
        finally:
            await subscriber.close()

    return EventSourceResponse(event_generator(), ping=15)