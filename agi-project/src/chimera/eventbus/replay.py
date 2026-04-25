"""Consume events from Kafka topics — for replay & offline analysis.

Two consumers are provided:

* :class:`KafkaTopicConsumer` — generic blocking iterator over a single topic.
* :class:`TraceReplayer` — reads ``chimera.agent_traces`` (and optionally
  related topics) and groups events by ``trace_id`` so a researcher can
  step through a past agent cycle.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from .config import EventBusConfig
from .events import ALL_TOPICS, Event, Topics

logger = logging.getLogger(__name__)


class KafkaTopicConsumer:
    """Blocking iterator that yields :class:`Event` objects from one or more topics.

    Example::

        cfg = EventBusConfig.from_env()
        with KafkaTopicConsumer(cfg, [Topics.AGENT_TRACES]) as consumer:
            for event in consumer:
                print(event.event_type, event.payload)
    """

    def __init__(
        self,
        config: EventBusConfig,
        topics: Sequence[str],
        *,
        group_id: str = "chimera-replay",
        auto_offset_reset: str = "earliest",
        consumer_timeout_ms: Optional[int] = None,
    ):
        if not config.bootstrap_servers:
            raise ValueError("KafkaTopicConsumer requires bootstrap_servers.")
        self.config = config
        self._topics = [config.topic(t) for t in topics]
        self._group_id = group_id
        self._auto_offset_reset = auto_offset_reset
        self._consumer_timeout_ms = consumer_timeout_ms
        self._consumer = None

    def _open(self):
        if self._consumer is not None:
            return self._consumer
        try:
            from kafka import KafkaConsumer  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "KafkaTopicConsumer requires 'kafka-python'. "
                "Install with: pip install kafka-python"
            ) from exc

        kwargs = dict(
            bootstrap_servers=self.config.bootstrap_servers,
            group_id=self._group_id,
            auto_offset_reset=self._auto_offset_reset,
            enable_auto_commit=True,
            value_deserializer=lambda v: v.decode("utf-8"),
        )
        if self._consumer_timeout_ms is not None:
            kwargs["consumer_timeout_ms"] = self._consumer_timeout_ms

        self._consumer = KafkaConsumer(*self._topics, **kwargs)
        return self._consumer

    def __enter__(self) -> "KafkaTopicConsumer":
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self) -> Iterator[Event]:
        consumer = self._open()
        for record in consumer:
            try:
                yield Event.from_json(record.value)
            except Exception as exc:
                logger.warning(
                    "Failed to deserialize event from %s: %s",
                    record.topic,
                    exc,
                )

    def close(self) -> None:
        if self._consumer is None:
            return
        try:
            self._consumer.close()
        except Exception:  # pragma: no cover - defensive
            pass
        self._consumer = None


class TraceReplayer:
    """Group events by ``trace_id`` so an agent cycle can be replayed.

    By default it subscribes to *all* Chimera topics. Pass ``topics=[...]``
    to restrict, or a list including only ``Topics.AGENT_TRACES`` if your
    agents only emit summary trace events.
    """

    def __init__(
        self,
        config: EventBusConfig,
        topics: Optional[Sequence[str]] = None,
        *,
        group_id: str = "chimera-trace-replay",
    ):
        self._consumer = KafkaTopicConsumer(
            config,
            topics if topics is not None else list(ALL_TOPICS),
            group_id=group_id,
        )

    def stream_traces(
        self, max_events: Optional[int] = None
    ) -> Iterator[Event]:
        """Yield events as they arrive."""
        with self._consumer as consumer:
            count = 0
            for event in consumer:
                yield event
                count += 1
                if max_events is not None and count >= max_events:
                    break

    def group_by_trace(
        self, max_events: int
    ) -> Dict[str, List[Event]]:
        """Read up to ``max_events`` events and group them by ``trace_id``.

        Events with no ``trace_id`` are bucketed under ``"_untraced"``.
        Within each bucket events stay in arrival order.
        """
        groups: Dict[str, List[Event]] = defaultdict(list)
        for event in self.stream_traces(max_events=max_events):
            key = event.trace_id or "_untraced"
            groups[key].append(event)
        return dict(groups)


def replay_from_iterable(events: Iterable[Event]) -> Dict[str, List[Event]]:
    """Group an arbitrary iterable of events by trace_id.

    Pure helper, used by tests and the InMemoryEventBus replay path.
    """
    groups: Dict[str, List[Event]] = defaultdict(list)
    for event in events:
        key = event.trace_id or "_untraced"
        groups[key].append(event)
    return dict(groups)


__all__ = [
    "KafkaTopicConsumer",
    "TraceReplayer",
    "replay_from_iterable",
    "Topics",
]
