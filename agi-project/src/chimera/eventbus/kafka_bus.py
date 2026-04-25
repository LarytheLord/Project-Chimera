"""Kafka-backed implementation of the Chimera event bus.

Uses ``kafka-python`` (pure Python, no librdkafka dependency) so the bus
works on any platform Chimera already targets. ``confluent-kafka`` could
be substituted later with no API changes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .bus import EventBus
from .config import EventBusConfig
from .events import Event, new_event

logger = logging.getLogger(__name__)


class KafkaEventBus(EventBus):
    """Publishes events to a Kafka / Redpanda cluster.

    The producer is created lazily on first ``publish`` so that constructing a
    bus does not require the broker to be reachable (handy for tests that
    only assert configuration).
    """

    def __init__(self, config: EventBusConfig):
        if not config.bootstrap_servers:
            raise ValueError(
                "KafkaEventBus requires at least one bootstrap server. "
                "Set CHIMERA_KAFKA_BROKERS or pass EventBusConfig(bootstrap_servers=[...])."
            )
        self.config = config
        self._producer = None  # lazy

    def _get_producer(self):
        if self._producer is not None:
            return self._producer
        try:
            from kafka import KafkaProducer  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "KafkaEventBus requires the 'kafka-python' package. "
                "Install with: pip install kafka-python"
            ) from exc

        self._producer = KafkaProducer(
            bootstrap_servers=self.config.bootstrap_servers,
            client_id=self.config.client_id,
            acks=self.config.acks,
            linger_ms=self.config.linger_ms,
            request_timeout_ms=self.config.request_timeout_ms,
            max_block_ms=self.config.max_block_ms,
            value_serializer=lambda v: v.encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k is not None else None,
        )
        return self._producer

    def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        *,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        event_type: Optional[str] = None,
        key: Optional[str] = None,
    ) -> Event:
        event = new_event(
            topic=topic,
            payload=payload,
            trace_id=trace_id,
            session_id=session_id or self.config.session_id,
            agent_id=agent_id or self.config.agent_id,
            event_type=event_type,
        )
        full_topic = self.config.topic(topic)
        # Partition key: prefer trace_id so all events from one cycle land on
        # the same partition (preserves per-trace ordering for replay).
        message_key = key or trace_id or event.session_id or event.event_id
        try:
            producer = self._get_producer()
            producer.send(full_topic, key=message_key, value=event.to_json())
        except Exception as exc:
            # Kafka failures must not break the agent loop. Log and move on.
            logger.warning(
                "KafkaEventBus.publish failed for topic %s: %s", full_topic, exc
            )
        return event

    def flush(self, timeout: Optional[float] = None) -> None:
        if self._producer is None:
            return
        try:
            self._producer.flush(timeout=timeout)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("KafkaEventBus.flush failed: %s", exc)

    def close(self) -> None:
        if self._producer is None:
            return
        try:
            self._producer.flush(timeout=2)
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            self._producer.close(timeout=2)
        except Exception:  # pragma: no cover - defensive
            pass
        self._producer = None
