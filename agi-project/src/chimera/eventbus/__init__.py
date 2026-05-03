"""
chimera.eventbus
================

Durable event bus for Project Chimera. Each stage of the agent loop
(perception, think, tool call, memory write, metacognitive reflection)
publishes a structured event so modules can evolve independently and so
agent traces can be replayed for debugging and research.

The bus is pluggable: use ``KafkaEventBus`` against a real broker (or
Redpanda) in production / development, ``InMemoryEventBus`` in tests, or
``NullEventBus`` when events should be silently dropped.

Quick start::

    from chimera.eventbus import build_event_bus, Topics

    bus = build_event_bus()  # reads CHIMERA_KAFKA_* env vars
    bus.publish(Topics.PERCEPTION, {"task": "hello"}, trace_id="t1")
    bus.close()
"""

from .bus import EventBus, InMemoryEventBus, NullEventBus
from .config import EventBusConfig
from .events import Event, Topics, ALL_TOPICS, new_event
from .factory import build_event_bus

__all__ = [
    "EventBus",
    "NullEventBus",
    "InMemoryEventBus",
    "EventBusConfig",
    "Event",
    "Topics",
    "ALL_TOPICS",
    "new_event",
    "build_event_bus",
]
