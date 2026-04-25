"""Event types and topic constants for the Chimera event bus."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


SCHEMA_VERSION = 1


class Topics:
    """Canonical Kafka topic names. Use these constants — never raw strings."""

    PERCEPTION = "chimera.perception"
    ACTIONS = "chimera.actions"
    TOOL_CALLS = "chimera.tool_calls"
    MEMORY_WRITES = "chimera.memory_writes"
    METACOG_REFLECTIONS = "chimera.metacog_reflections"
    AGENT_TRACES = "chimera.agent_traces"


ALL_TOPICS = (
    Topics.PERCEPTION,
    Topics.ACTIONS,
    Topics.TOOL_CALLS,
    Topics.MEMORY_WRITES,
    Topics.METACOG_REFLECTIONS,
    Topics.AGENT_TRACES,
)


@dataclass
class Event:
    """A single event flowing through the bus.

    ``trace_id`` is shared by all events emitted in one perceive→think→act
    cycle so a downstream consumer can reconstruct the cycle.

    ``session_id`` groups multiple cycles produced by a single agent run.
    """

    topic: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    event_type: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=_json_default, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        return cls(
            topic=data["topic"],
            payload=data.get("payload", {}),
            event_id=data.get("event_id", str(uuid.uuid4())),
            trace_id=data.get("trace_id"),
            session_id=data.get("session_id"),
            agent_id=data.get("agent_id"),
            event_type=data.get("event_type"),
            timestamp=data.get("timestamp", time.time()),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, raw: str) -> "Event":
        return cls.from_dict(json.loads(raw))


def new_event(
    topic: str,
    payload: Dict[str, Any],
    *,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    event_type: Optional[str] = None,
) -> Event:
    """Convenience constructor that fills in event_id and timestamp."""
    return Event(
        topic=topic,
        payload=payload,
        trace_id=trace_id,
        session_id=session_id,
        agent_id=agent_id,
        event_type=event_type,
    )


def _json_default(obj: Any) -> Any:
    """Best-effort JSON fallback for non-serializable values in payloads.

    Agent payloads carry arbitrary observations / outcomes from tools, so we
    coerce unknown types to ``str`` rather than crash the producer.
    """
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(obj, "_asdict"):  # NamedTuple
        return obj._asdict()
    return str(obj)
