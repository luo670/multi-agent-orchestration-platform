"""Celery application instance and configuration.

Kept separate from ``tasks.py`` so that the worker can be started with a
clean import path and the broker URL is centrally configured from env vars.

Start the worker:
    celery -A app.celery_app worker --loglevel=info

The worker uses the **prefork** pool (default). Celery tasks that need to run
async code wrap the coroutine in ``asyncio.run()`` inside the task body — this
is the simplest, most reliable approach and avoids gevent/eventlet pitfalls
with asyncio-based libraries (LangGraph, SQLAlchemy async, redis.asyncio).
"""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "agent_platform",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks"],
)

# ---- Tuning ----
celery_app.conf.update(
    # Serialize with JSON (orjson via Celery's json hook if installed).
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="Asia/Shanghai",
    enable_utc=True,
    # Reliability: acknowledge AFTER the task completes (not before).
    task_acks_late=True,
    # Requeue a task if the worker crashes mid-execution.
    task_reject_on_worker_lost=True,
    # Concurrency: one task at a time per worker process by default.
    # Scale horizontally by running more worker processes/containers.
    worker_prefetch_multiplier=1,
    # Visibility timeout — keep a task reserved long enough for multi-step
    # agent pipelines (default 1h is too short for long LLM chains).
    broker_transport_options={"visibility_timeout": 3600},
    # Result expiry — keep results for 6 hours.
    result_expires=6 * 3600,
)


@celery_app.task(name="health_check")
def health_check() -> str:
    """Trivial task for uptime/smoke checks."""
    return "ok"