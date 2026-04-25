"""Configuration for the Chimera event bus."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _split_brokers(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [b.strip() for b in value.split(",") if b.strip()]


@dataclass
class EventBusConfig:
    """Event bus configuration.

    ``bootstrap_servers`` is the Kafka / Redpanda broker list. When empty,
    :func:`chimera.eventbus.build_event_bus` returns a ``NullEventBus`` so
    Chimera continues to work without a broker.
    """

    bootstrap_servers: List[str] = field(default_factory=list)
    client_id: str = "chimera-agent"
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    topic_prefix: str = ""
    enabled: bool = True
    # Producer tuning
    acks: str = "all"
    linger_ms: int = 5
    request_timeout_ms: int = 10_000
    max_block_ms: int = 5_000

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "EventBusConfig":
        env = env if env is not None else os.environ
        enabled_raw = env.get("CHIMERA_EVENTBUS_ENABLED", "1").lower()
        return cls(
            bootstrap_servers=_split_brokers(env.get("CHIMERA_KAFKA_BROKERS")),
            client_id=env.get("CHIMERA_KAFKA_CLIENT_ID", "chimera-agent"),
            agent_id=env.get("CHIMERA_AGENT_ID"),
            session_id=env.get("CHIMERA_SESSION_ID"),
            topic_prefix=env.get("CHIMERA_KAFKA_TOPIC_PREFIX", ""),
            enabled=enabled_raw not in ("0", "false", "no", ""),
        )

    def topic(self, name: str) -> str:
        if self.topic_prefix:
            return f"{self.topic_prefix}{name}"
        return name
