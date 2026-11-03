"""SQLAlchemy ORM models.

Two tables:
    - ``tasks``      : business workflow table — one row per article generation request.
    - ``agent_metrics``: observability table — one row per LLM call (node-level telemetry).

The ``agent_metrics`` table is the backbone of the observability layer:
every LLM invocation (Researcher / Writer / Reviewer) records its node name,
duration, token consumption, and (for Reviewer) pass/fail result.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


# Reusable statuses (kept as plain strings for DB portability).
TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# Node names used across the system.
NODE_RESEARCHER = "researcher"
NODE_WRITER = "writer"
NODE_REVIEWER = "reviewer"


def _generate_uuid() -> str:
    """Generate a hex UUID4 string for task_id defaults."""
    return uuid.uuid4().hex


class Task(Base):
    """Business workflow table — tracks a single article-generation request.

    Mirrors the LangGraph State fields (topic, research_notes, draft, feedback,
    revision_count) so that the graph state can be persisted/restored via this
    table in addition to the LangGraph Checkpointer.
    """

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        default=_generate_uuid,
        comment="Public task identifier (UUID4 hex)",
    )
    topic: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="The tech topic submitted by the user",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default=TASK_STATUS_PENDING,
        index=True,
        comment="pending | running | completed | failed",
    )
    # LangGraph State snapshot fields
    research_notes: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Notes produced by the Researcher node"
    )
    draft: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Article draft produced by the Writer node"
    )
    feedback: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Latest feedback from the Reviewer node"
    )
    revision_count: Mapped[int] = mapped_column(
        Integer, default=0, comment="How many times the draft has been rewritten"
    )
    final_article: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Final approved article (status=completed)"
    )
    error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Error trace if status=failed"
    )
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationship: one task -> many metric rows
    metrics: Mapped[list[AgentMetric]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Task task_id={self.task_id!r} topic={self.topic!r} "
            f"status={self.status!r} revisions={self.revision_count}>"
        )


class AgentMetric(Base):
    """Observability table — one row per LLM call.

    Populated by the metrics interceptor decorator (utils.py) that wraps every
    node function. This data powers the observability dashboard and allows
    post-hoc analysis of per-node latency, token cost, and review pass rate.
    """

    __tablename__ = "agent_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tasks.task_id", ondelete="CASCADE"),
        index=True,
        comment="FK to tasks.task_id",
    )
    node_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="researcher | writer | reviewer",
    )
    duration_ms: Mapped[int] = mapped_column(
        Integer, default=0, comment="Wall-clock duration of this node call (ms)"
    )
    prompt_tokens: Mapped[int] = mapped_column(
        Integer, default=0, comment="Input tokens consumed"
    )
    completion_tokens: Mapped[int] = mapped_column(
        Integer, default=0, comment="Output tokens consumed"
    )
    total_tokens: Mapped[int] = mapped_column(
        Integer, default=0, comment="prompt_tokens + completion_tokens"
    )
    passed: Mapped[bool | None] = mapped_column(
        nullable=True,
        comment="Only set by Reviewer: True=approved, False=rejected. NULL for other nodes.",
    )
    feedback: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Reviewer feedback text (if any)"
    )
    model_name: Mapped[str | None] = mapped_column(
        String(128), nullable=True, comment="LLM model used for this call"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # Relationship back to parent task
    task: Mapped[Task] = relationship(back_populates="metrics")

    # Composite index for common query: "all metrics for a task, ordered by time"
    __table_args__ = (
        Index("idx_task_node", "task_id", "node_name"),
        Index("idx_created_at", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<AgentMetric task_id={self.task_id!r} node={self.node_name!r} "
            f"duration={self.duration_ms}ms tokens={self.total_tokens}>"
        )