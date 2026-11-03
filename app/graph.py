"""LangGraph core orchestration logic.

Defines:
    - AgentState (TypedDict): topic, research_notes, draft, feedback, revision_count
    - Researcher / Writer / Reviewer nodes with engineered prompts
    - Conditional routing (Reviewer -> Writer on reject, -> END on pass / max revisions)
    - Checkpointer (AsyncSqliteSaver) for state persistence & crash recovery

Hooks left for Step 4 (utils.py):
    - Every LLM call goes through ``_invoke_llm()``, which records a
      ``LLMCallRecord`` into ``config["configurable"]["metrics_collector"]``.
      Step 4's ``@with_metrics`` decorator manages the collector lifecycle
      and writes rows to ``agent_metrics``.
    - ``_invoke_llm()`` is a plain function that Step 4's ``@with_retry``
      decorator wraps with Tenacity exponential backoff.
    This file is fully runnable on its own; the decorators are enhancements.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict

from app.config import settings
from app.events import EventPublisher


# ============================================================
# 1. State Definition
# ============================================================

class AgentState(TypedDict, total=False):
    """Shared state flowing through the graph.

    All fields are optional at different points in the workflow:
      - Researcher sets  research_notes
      - Writer sets      draft (and bumps revision_count)
      - Reviewer sets    feedback (+ passed via the routing decision)
    """

    topic: str
    research_notes: str
    draft: str
    feedback: str
    revision_count: int


# ============================================================
# 2. Structured output schema for the Reviewer
# ============================================================

class ReviewResult(BaseModel):
    """Structured review outcome produced by the Reviewer node."""

    passed: bool = Field(description="True if the draft meets quality bar")
    feedback: str = Field(description="Specific, actionable feedback; required when passed is False")


# ============================================================
# 3. LLM call bookkeeping (consumed by Step 4 decorators)
# ============================================================

@dataclass
class LLMCallRecord:
    """Single LLM invocation telemetry. Collected per node, flushed to DB."""

    task_id: str
    node_name: str
    duration_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_name: str = ""
    passed: bool | None = None
    feedback: str = ""
    started_at: float = field(default_factory=time.time)


def _build_llm(temperature: float = 0.7) -> ChatOpenAI:
    """Construct a ChatOpenAI client pointed at the DashScope endpoint.

    All credentials come from environment variables (settings) — NOTHING
    is hardcoded. Switching models only requires changing QWEN_MODEL in .env.
    """
    return ChatOpenAI(
        model=settings.QWEN_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,  # from env, never literal
        base_url=settings.DASHSCOPE_BASE_URL,
        temperature=temperature,
        streaming=False,
    )


async def _invoke_llm(
    llm: ChatOpenAI,
    messages: list,
    *,
    task_id: str,
    node_name: str,
    config: dict[str, Any],
) -> str:
    """Invoke the LLM and collect telemetry.

    This is the single chokepoint for every model call in the graph.
    Step 4's ``@with_retry`` wraps this with Tenacity; Step 4's
    ``@with_metrics`` provides the ``metrics_collector`` in config.

    Returns the raw text content of the response.
    """
    started = time.perf_counter()
    response = await llm.ainvoke(messages)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # Extract token usage if the provider returns it.
    usage = getattr(response, "usage_metadata", None) or {}
    prompt_tokens = int(usage.get("input_tokens", 0))
    completion_tokens = int(usage.get("output_tokens", 0))

    record = LLMCallRecord(
        task_id=task_id,
        node_name=node_name,
        duration_ms=elapsed_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        model_name=settings.QWEN_MODEL,
    )

    # Hand the record to the collector (provided by @with_metrics in Step 4).
    collector = config.get("configurable", {}).get("metrics_collector")
    if collector is not None:
        collector.append(record)

    return response.content


# ============================================================
# 4. Helper: read runtime values from config
# ============================================================

def _ctx(config: dict[str, Any]) -> tuple[str, EventPublisher | None]:
    """Extract task_id and optional publisher from LangGraph runtime config."""
    conf = config.get("configurable", {})
    return conf.get("task_id", "unknown"), conf.get("publisher")


# ============================================================
# 5. Node: Researcher
# ============================================================

_RESEARCHER_SYSTEM = """You are a senior technology researcher for a tech media editorial team.
Given a topic, produce a thorough, well-structured research note covering:
  1. Background & definition of the topic
  2. Current state of the art / latest developments
  3. Key players, products, or open-source projects
  4. Technical principles (concise, accurate)
  5. Industry impact & future outlook
  6. 3-5 credible reference points (name the source type, no need for real URLs)

Be factual, precise, and information-dense. Use Markdown headings.
Output ONLY the research note, no preamble."""


async def researcher_node(state: AgentState, config: dict[str, Any]) -> dict[str, Any]:
    """Research agent: generates structured research notes on the topic."""
    task_id, publisher = _ctx(config)

    if publisher:
        await publisher.publish(task_id, "node_start", {"node": "researcher", "topic": state["topic"]})

    llm = _build_llm(temperature=0.3)
    messages = [
        SystemMessage(content=_RESEARCHER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}"),
    ]
    notes = await _invoke_llm(
        llm, messages, task_id=task_id, node_name="researcher", config=config
    )

    if publisher:
        await publisher.publish(
            task_id, "node_end", {"node": "researcher", "chars": len(notes)}
        )

    return {"research_notes": notes, "revision_count": 0}


# ============================================================
# 6. Node: Writer
# ============================================================

_WRITER_SYSTEM = """You are a senior technology writer for a tech media editorial team.
You write engaging, accurate, well-structured articles for a technical audience.

If REVISION FEEDBACK is provided, it means your previous draft was rejected.
You MUST revise the draft to address EVERY point in the feedback while keeping
the parts that worked."""

_WRITER_TEMPLATE = """Topic: {topic}

=== RESEARCH NOTES ===
{research_notes}

=== REVISION FEEDBACK (if any) ===
{feedback}
=== END FEEDBACK ===

Write a polished article (1200-2000 Chinese characters) with:
  - A compelling title (as an H1)
  - A strong lede that hooks the reader
  - 3-5 sections with clear H2 headings
  - Accurate technical detail grounded in the research notes
  - A concise conclusion

Output ONLY the article in Markdown."""


async def writer_node(state: AgentState, config: dict[str, Any]) -> dict[str, Any]:
    """Writing agent: produces a draft from research notes (+ feedback on revision)."""
    task_id, publisher = _ctx(config)

    feedback = state.get("feedback", "")
    is_revision = state.get("revision_count", 0) > 0

    if publisher:
        await publisher.publish(
            task_id,
            "node_start",
            {"node": "writer", "revision": is_revision, "attempt": state.get("revision_count", 0) + 1},
        )

    llm = _build_llm(temperature=0.7)
    user_msg = _WRITER_TEMPLATE.format(
        topic=state["topic"],
        research_notes=state.get("research_notes", ""),
        feedback=feedback if feedback else "(none — this is the first draft)",
    )
    messages = [SystemMessage(content=_WRITER_SYSTEM), HumanMessage(content=user_msg)]
    draft = await _invoke_llm(
        llm, messages, task_id=task_id, node_name="writer", config=config
    )

    new_count = state.get("revision_count", 0) + 1

    if publisher:
        await publisher.publish(task_id, "node_end", {"node": "writer", "chars": len(draft)})

    return {"draft": draft, "revision_count": new_count}


# ============================================================
# 7. Node: Reviewer
# ============================================================

_REVIEWER_SYSTEM = """You are a strict but fair editor for a tech media editorial team.
Evaluate the submitted draft against these criteria:
  1. Technical accuracy
  2. Logical structure & readability
  3. Engagement & writing quality
  4. Faithfulness to the research notes (no fabrication)
  5. Length appropriateness (1200-2000 Chinese chars)

Respond with STRICT JSON only, no markdown fences, no commentary:
{"passed": <true|false>, "feedback": "<if rejected, list every problem concretely; if passed, a one-line commendation>"}

Reject anything that is factually loose, structurally weak, or padded with filler."""


async def reviewer_node(state: AgentState, config: dict[str, Any]) -> dict[str, Any]:
    """Review agent: returns a pass/fail decision with actionable feedback."""
    task_id, publisher = _ctx(config)

    if publisher:
        await publisher.publish(task_id, "node_start", {"node": "reviewer", "attempt": state.get("revision_count", 0)})

    llm = _build_llm(temperature=0.1)
    messages = [
        SystemMessage(content=_REVIEWER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}\n\n=== DRAFT ===\n{state.get('draft', '')}"),
    ]
    raw = await _invoke_llm(
        llm, messages, task_id=task_id, node_name="reviewer", config=config
    )

    review = _parse_review(raw)

    if publisher:
        await publisher.publish(
            task_id,
            "review_result",
            {"passed": review.passed, "feedback": review.feedback, "attempt": state.get("revision_count", 0)},
        )

    return {"feedback": review.feedback, "passed": review.passed}


def _parse_review(raw: str) -> ReviewResult:
    """Robustly extract a ReviewResult from the LLM output.

    Handles common failure modes: markdown code fences, surrounding prose,
    or partial JSON. Falls back to a reject with the raw text as feedback.
    """
    text = raw.strip()

    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    # Otherwise grab the first {...} block.
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        text = brace.group(0)

    try:
        data = json.loads(text)
        return ReviewResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        # Safer to reject than to ship a bad article.
        return ReviewResult(passed=False, feedback=f"Reviewer output was not valid JSON. Raw: {raw[:500]}")


# ============================================================
# 8. Conditional routing
# ============================================================

def route_after_review(state: AgentState) -> str:
    """Decide the next node after the Reviewer.

    Returns "writer" to rewrite, or END to finish.

    Finish conditions:
      - Reviewer approved (passed == True)
      - Revision count exceeded MAX_REVISIONS (force-complete to avoid infinite loops)
    """
    revision_count = state.get("revision_count", 0)

    # Check reviewer's verdict stored in state by reviewer_node.
    # ``passed`` is not part of AgentState TypedDict (it's a routing signal),
    # so we read it defensively.
    passed = state.get("passed", False)  # type: ignore[typeddict-item]

    if passed:
        return END
    if revision_count >= settings.MAX_REVISIONS:
        # Force-complete: too many revisions, ship the latest draft.
        return END
    return "writer"


# ============================================================
# 9. Graph builder + Checkpointer
# ============================================================

# Module-level singletons for the compiled graph and checkpointer.
_compiled_graph: Any = None
_checkpointer: AsyncSqliteSaver | None = None
_checkpointer_conn: Any = None


def build_graph() -> StateGraph:
    """Construct the (uncompiled) StateGraph with nodes and edges.

    Topology:
        START -> researcher -> writer -> reviewer --(route_after_review)--> {
            writer     (rejected, revisions remain)
            END        (approved OR max revisions reached)
        }
    """
    graph = StateGraph(AgentState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    # Linear start.
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")

    # Conditional loop on the reviewer's verdict.
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "writer": "writer",
            END: END,
        },
    )

    return graph


async def get_compiled_graph() -> Any:
    """Return a singleton compiled graph with a persistent SQLite checkpointer.

    The checkpointer writes to ``checkpoints.db`` so that if a Celery worker
    crashes mid-task, the graph state can be resumed on restart using the
    same ``thread_id`` (== task_id).
    """
    global _compiled_graph, _checkpointer, _checkpointer_conn

    if _compiled_graph is not None:
        return _compiled_graph

    import aiosqlite

    # Persistent file — survives restarts.
    _checkpointer_conn = await aiosqlite.connect("checkpoints.db")
    _checkpointer = AsyncSqliteSaver(_checkpointer_conn)
    await _checkpointer.setup()  # create checkpoint tables

    graph = build_graph()
    _compiled_graph = graph.compile(checkpointer=_checkpointer)
    return _compiled_graph


async def close_graph() -> None:
    """Close the checkpointer connection. Call on application shutdown."""
    global _compiled_graph, _checkpointer, _checkpointer_conn
    if _checkpointer_conn is not None:
        await _checkpointer_conn.close()
    _compiled_graph = None
    _checkpointer = None
    _checkpointer_conn = None


# ============================================================
# 10. High-level entrypoint (used by tasks.py in Step 3)
# ============================================================

async def run_pipeline(
    task_id: str,
    topic: str,
    publisher: EventPublisher | None = None,
) -> dict[str, Any]:
    """Run the full agent pipeline for a single task.

    Args:
        task_id:  Unique task identifier; also used as the LangGraph thread_id
                  so the checkpointer can resume on crash.
        topic:    The tech topic to write about.
        publisher: Optional EventPublisher for real-time SSE events.

    Returns:
        The final state dict (contains draft, feedback, revision_count, etc.).
    """
    app = await get_compiled_graph()
    initial_state: AgentState = {"topic": topic}  # type: ignore[typeddict-item]
    run_config = {
        "configurable": {
            "thread_id": task_id,
            "task_id": task_id,
            "publisher": publisher,
            "metrics_collector": [],  # Step 4's @with_metrics populates this
        }
    }
    final_state = await app.ainvoke(initial_state, config=run_config)
    return final_state