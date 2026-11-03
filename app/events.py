"""Real-time event publisher based on Redis Pub/Sub.

Architecture:
    Celery Worker (runs the graph)
        └─ EventPublisher.publish(task_id, ...)
              └─→ Redis Pub/Sub channel: "task:{task_id}:events"
                        ▲
    FastAPI SSE endpoint (GET /tasks/{task_id}/stream)
        └─ subscribes to the same channel, forwards events to the client

This decouples the producer (Celery worker) from the consumer (FastAPI SSE),
so multiple SSE clients can listen to the same task and the worker never blocks
on streaming.
"""

from __future__ import annotations

import json
import time
from typing import Any

import redis.asyncio as aioredis

from app.config import settings


def _channel(task_id: str) -> str:
    """Redis channel name for a given task's events."""
    return f"task:{task_id}:events"


class EventPublisher:
    """Publishes node-progress events to Redis.

    Intended to be passed into the LangGraph runtime config
    (``config["configurable"]["publisher"]``) so node functions can emit
    real-time status updates without knowing about the transport layer.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_url = redis_url or settings.REDIS_URL
        self._redis: aioredis.Redis | None = None

    async def _ensure(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    async def publish(
        self,
        task_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Publish an event. Failures are swallowed so the graph never breaks.

        event_type examples:
            - "node_start"   ({"node": "researcher", "topic": "..."})
            - "node_end"     ({"node": "researcher", "summary": "..."})
            - "revision"     ({"attempt": 2, "max": 3})
            - "task_done"    ({"final_article": "..."})
            - "task_failed"  ({"error": "..."})
        """
        payload = {
            "type": event_type,
            "data": data or {},
            "ts": time.time(),
        }
        try:
            r = await self._ensure()
            await r.publish(_channel(task_id), json.dumps(payload, ensure_ascii=False))
        except Exception:  # noqa: BLE001 — telemetry must never break the pipeline
            # In a real system you'd log this; here we keep the graph resilient.
            pass

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None


class EventSubscriber:
    """Subscribes to a task's event channel.

    Used by the FastAPI SSE endpoint to stream events to the client.
    Yields raw JSON strings ready to be sent as SSE ``data:`` payloads.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_url = redis_url or settings.REDIS_URL
        self._pubsub: aioredis.client.PubSub | None = None
        self._redis: aioredis.Redis | None = None

    async def listen(self, task_id: str):
        """Async generator yielding event JSON strings for a task.

        Yields until the connection is closed or the channel signals
        a terminal event ("task_done" / "task_failed"). After a terminal
        event the generator stops, allowing the SSE endpoint to close.
        """
        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(_channel(task_id))

        try:
            async for message in self._pubsub.listen():
                if message["type"] != "message":
                    continue
                raw: str = message["data"]
                yield raw

                # Detect terminal events to end the SSE stream cleanly.
                if '"task_done"' in raw or '"task_failed"' in raw:
                    break
        finally:
            await self.close()

    async def close(self) -> None:
        if self._pubsub is not None:
            await self._pubsub.unsubscribe()
            await self._pubsub.aclose()
            self._pubsub = None
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None