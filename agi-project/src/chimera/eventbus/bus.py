"""Abstract event bus + in-process implementations."""

from __future__ import annotations

import abc
import threading
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Iterable, List, Optional

from .events import Event, new_event


class EventBus(abc.ABC):
    """Abstract publisher/subscriber bus.

    Implementations only need to support fire-and-forget ``publish``.
    Consumers (``subscribe``, ``drain``) are optional — the Kafka
    implementation provides them via the dedicated consumer in
    :mod:`chimera.eventbus.replay`.
    """

    @abc.abstractmethod
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
        """Publish a payload to ``topic`` and return the constructed Event."""

    def flush(self, timeout: Optional[float] = None) -> None:
        """Block until pending events are delivered. No-op by default."""

    def close(self) -> None:
        """Release any underlying resources. No-op by default."""


class NullEventBus(EventBus):
    """Drops every event. Used when Kafka is unconfigured."""

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
        return new_event(
            topic=topic,
            payload=payload,
            trace_id=trace_id,
            session_id=session_id,
            agent_id=agent_id,
            event_type=event_type,
        )


Subscriber = Callable[[Event], None]


class InMemoryEventBus(EventBus):
    """Thread-safe in-process bus.

    Useful for tests and for single-process Chimera deployments that want
    pub/sub semantics without standing up a broker. Events are retained in a
    bounded per-topic deque so consumers that join late can still drain
    recent history.
    """

    def __init__(self, history_per_topic: int = 1000):
        self._lock = threading.RLock()
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_per_topic)
        )
        self._subscribers: Dict[str, List[Subscriber]] = defaultdict(list)

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
            session_id=session_id,
            agent_id=agent_id,
            event_type=event_type,
        )
        with self._lock:
            self._history[topic].append(event)
            subs = list(self._subscribers.get(topic, ()))
        for callback in subs:
            callback(event)
        return event

    def subscribe(self, topic: str, callback: Subscriber) -> None:
        with self._lock:
            self._subscribers[topic].append(callback)

    def drain(self, topic: str) -> List[Event]:
        with self._lock:
            events = list(self._history.get(topic, ()))
            self._history[topic].clear()
        return events

    def history(self, topic: str) -> List[Event]:
        with self._lock:
            return list(self._history.get(topic, ()))

    def topics(self) -> Iterable[str]:
        with self._lock:
            return list(self._history.keys())
