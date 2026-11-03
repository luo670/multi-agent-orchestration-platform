"""Cross-cutting infrastructure: metrics interceptor, retry, rate limiting.

Three concerns, three tools:

    1. @with_metrics  — decorator for graph node functions.
       Wraps a node so that every LLMCallRecord produced inside it is flushed
       to the ``agent_metrics`` table after the node returns. The collector
       list is injected into the LangGraph runtime config.

    2. @with_retry    — decorator for the _invoke_llm chokepoint in graph.py.
       Tenacity exponential backoff for transient LLM API errors (429, 5xx,
       timeouts, connection resets). Non-retryable errors (auth, bad request)
       propagate immediately.

    3. RedisRateLimiter — distributed concurrency limiter.
       Uses a Redis SET NX + EX lease to enforce a global max-concurrent-LLM-
       calls cap. Prevents token-rate-limit (429) storms under high load.

Wiring into the graph:
    - graph.py's node functions get @with_metrics applied at graph build time
      (see the ``build_graph()`` patch below, or apply manually).
    - graph.py's ``_invoke_llm`` gets @with_retry applied as the chokepoint.
    - RedisRateLimiter is optional; wrap a node or _invoke_llm with it when
      you need hard concurrency caps (e.g. free-tier API plans).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

import redis.asyncio as aioredis
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.config import settings
from app.database import AsyncSessionLocal
from app.graph import LLMCallRecord
from app.models import AgentMetric

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# ============================================================
# 1. Metrics interceptor — @with_metrics
# ============================================================

def with_metrics(
    node_fn: Callable[..., Awaitable[dict[str, Any]]],
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Decorate a LangGraph node function to persist telemetry to agent_metrics.

    Lifecycle:
      1. Read (or create) the metrics_collector list from runtime config.
      2. Snapshot its length before the node runs.
      3. Run the node (which internally calls _invoke_llm → appends records).
      4. After the node returns, flush the NEW records to agent_metrics.
         (Records produced by this node = collector[before_len:].)

    This keeps the collector decoupled: graph.py appends raw records without
    knowing about the DB; this decorator is the single persistence point.

    For the Reviewer node, if the returned state contains ``passed``, that
    boolean is stamped onto the reviewer's metrics record before flushing.
    """

    @functools.wraps(node_fn)
    async def wrapper(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        conf = config.setdefault("configurable", {})
        collector: list[LLMCallRecord] = conf.setdefault("metrics_collector", [])
        task_id = conf.get("task_id", "unknown")

        before_len = len(collector)
        node_name = getattr(node_fn, "__name__", "unknown")

        logger.debug("with_metrics: node=%s task=%s starting", node_name, task_id)

        # Execute the node.
        result = await node_fn(state, config)

        # Collect only the records produced by THIS node call.
        new_records = collector[before_len:]

        # Special-case: stamp reviewer pass/fail onto its record.
        if node_name == "reviewer_node" and "passed" in result:
            passed_val = result.get("passed")
            for rec in new_records:
                if rec.node_name == "reviewer":
                    rec.passed = bool(passed_val)
                    rec.feedback = result.get("feedback", "")

        # Persist to DB (best-effort — never let metrics break the pipeline).
        if new_records:
            try:
                await _flush_metrics(new_records)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to flush metrics for task=%s node=%s", task_id, node_name)

        return result

    return wrapper


async def _flush_metrics(records: list[LLMCallRecord]) -> None:
    """Batch-insert a list of LLMCallRecord into the agent_metrics table."""
    if not records:
        return
    rows = [
        AgentMetric(
            task_id=r.task_id,
            node_name=r.node_name,
            duration_ms=r.duration_ms,
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
            total_tokens=r.total_tokens,
            passed=r.passed,
            feedback=r.feedback,
            model_name=r.model_name,
        )
        for r in records
    ]
    async with AsyncSessionLocal() as session:
        session.add_all(rows)
        await session.commit()
    logger.info("Flushed %d metric rows", len(rows))


def apply_metrics_to_graph() -> None:
    """Monkey-patch graph.py's node functions with @with_metrics.

    Call this once at app startup (before the first pipeline runs).
    Alternatively, apply @with_metrics directly to the node function
    definitions in graph.py — that's cleaner but couples graph.py to
    utils.py at import time, which can cause circular imports. The patch
    approach keeps graph.py importable standalone.

    Why patch instead of wrap at build_graph()?
      build_graph() calls graph.add_node(name, fn). We wrap the fn *before*
      build_graph runs, so the compiled graph calls our wrapped version.
    """
    import app.graph as g

    g.researcher_node = with_metrics(g.researcher_node)
    g.writer_node = with_metrics(g.writer_node)
    g.reviewer_node = with_metrics(g.reviewer_node)
    logger.info("Metrics interceptor applied to graph nodes")


# ============================================================
# 2. Retry — @with_retry (Tenacity exponential backoff)
# ============================================================

# Exception types that indicate a TRANSIENT failure worth retrying.
# Localization kept loose on purpose — DashScope / OpenAI-compatible
# endpoints surface rate limits and server errors in various forms.
_RETRYABLE_HINTS = (
    "429",            # rate limit
    "500", "502", "503", "504",  # server errors
    "timeout", "timed out",
    "connection reset",
    "connection error",
    "temporary",
    "overloaded",
)


def _is_retryable(exc: BaseException) -> bool:
    """Classify an exception as retryable based on its string representation.

    We avoid importing SDK-specific exception classes (which vary between
    openai versions) by matching on the error message. This is pragmatic and
    robust across SDK upgrades.
    """
    msg = str(exc).lower()
    return any(hint in msg for hint in _RETRYABLE_HINTS)


class _RetryableError(Exception):
    """Marker wrapper so Tenacity's retry_if_exception_type works cleanly."""


def with_retry(
    max_attempts: int | None = None,
    base_wait: float = 1.0,
    max_wait: float = 30.0,
):
    """Decorator: Tenacity exponential backoff for an async function.

    Default max_attempts comes from settings.MAX_RETRIES (env: MAX_RETRIES).

    Backoff: wait_exponential(multiplier=base_wait, max=max_wait)
      attempt 1 fail → wait ~1s
      attempt 2 fail → wait ~2s
      attempt 3 fail → wait ~4s
      ...capped at max_wait

    Only retries transient errors (429/5xx/timeout/connection). Auth errors
    (401/403) and bad-request (400) propagate immediately — retrying them
    is pointless and burns time.
    """
    attempts = max_attempts or settings.MAX_RETRIES

    def decorator(fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(fn)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if not _is_retryable(exc) or attempt == attempts:
                        raise
                    # Compute exponential backoff with jitter.
                    wait = min(max_wait, base_wait * (2 ** (attempt - 1)))
                    logger.warning(
                        "with_retry: attempt %d/%d failed (%s). Retrying in %.1fs",
                        attempt, attempts, type(exc).__name__, wait,
                    )
                    await asyncio.sleep(wait)
            # Unreachable, but keeps mypy happy.
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("with_retry: exhausted without exception")

        return async_wrapper

    return decorator


# ============================================================
# 3. Distributed rate limiter — RedisRateLimiter
# ============================================================

class RedisRateLimiter:
    """Distributed concurrency limiter using Redis leases.

    Enforces a global cap on concurrent LLM calls (across all Celery workers).
    Uses ``SET NX EX`` to acquire a lease; on expiry the slot auto-releases
    so a crashed worker doesn't leave the lock stuck.

    Usage:
        limiter = RedisRateLimiter(max_concurrent=5, ttl=120)
        async with limiter:
            response = await llm.ainvoke(...)
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        ttl: int = 120,
        redis_url: str | None = None,
        key_prefix: str = "llm:concurrencyslot",
    ) -> None:
        self.max_concurrent = max_concurrent
        self.ttl = ttl
        self._redis_url = redis_url or settings.REDIS_URL
        self._key_prefix = key_prefix
        self._redis: aioredis.Redis | None = None
        self._acquired_key: str | None = None
        self._acquire_timeout = 60.0  # max seconds to wait for a slot
        self._poll_interval = 0.3

    async def __aenter__(self) -> "RedisRateLimiter":
        await self._acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._release()

    async def _ensure_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    async def _acquire(self) -> None:
        """Try to claim one of ``max_concurrent`` slots. Block until acquired."""
        r = await self._ensure_redis()
        deadline = time.monotonic() + self._acquire_timeout

        while time.monotonic() < deadline:
            for slot in range(self.max_concurrent):
                key = f"{self._key_prefix}:{slot}"
                token = uuid.uuid4().hex
                # SET key token NX EX  →  returns OK if acquired.
                got = await r.set(key, token, nx=True, ex=self.ttl)
                if got:
                    self._acquired_key = key
                    self._acquired_token = token
                    logger.debug("RateLimiter: acquired slot %d", slot)
                    return
            # No slot free — back off briefly and retry.
            await asyncio.sleep(self._poll_interval)

        # Timed out waiting for a slot.
        raise TimeoutError(
            f"RedisRateLimiter: no free slot within {self._acquire_timeout}s "
            f"(max_concurrent={self.max_concurrent})"
        )

    async def _release(self) -> None:
        """Release the held slot. Uses a Lua-style CAS to avoid releasing
        a slot that was auto-expired and re-acquired by another worker.
        """
        if self._acquired_key is None or self._redis is None:
            return
        # Atomic check-and-delete via pipeline + watch (avoids deleting a
        # slot that expired and was taken by someone else).
        r = self._redis
        key = self._acquired_key
        token = getattr(self, "_acquired_token", "")
        try:
            # Lua script: only delete if value matches our token.
            lua = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('del', KEYS[1])
            else
                return 0
            end
            """
            await r.eval(lua, 1, key, token)
        except Exception:  # noqa: BLE001
            logger.exception("RateLimiter: release failed for key=%s", key)
        finally:
            self._acquired_key = None
            self._acquired_token = None

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None


# ============================================================
# 4. Convenience: wire retry into the LLM chokepoint
# ============================================================

def install_retry_on_invoke_llm() -> None:
    """Wrap graph.py's _invoke_llm with @with_retry.

    Call this once at startup (after apply_metrics_to_graph()). This keeps
    graph.py importable without a tenacity dependency, and lets you swap
    retry strategies without touching graph.py.
    """
    import app.graph as g

    original = g._invoke_llm
    if getattr(original, "_retry_wrapped", False):
        return  # idempotent

    wrapped = with_retry()(original)
    wrapped._retry_wrapped = True  # type: ignore[attr-defined]
    g._invoke_llm = wrapped
    logger.info("Retry decorator installed on graph._invoke_llm")


def install_all_infra() -> None:
    """One-call setup: apply metrics + retry to the graph at app startup."""
    apply_metrics_to_graph()
    install_retry_on_invoke_llm()