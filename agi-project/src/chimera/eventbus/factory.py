"""Factory for selecting the right :class:`EventBus` implementation."""

from __future__ import annotations

import logging
from typing import Optional

from .bus import EventBus, NullEventBus
from .config import EventBusConfig

logger = logging.getLogger(__name__)


def build_event_bus(config: Optional[EventBusConfig] = None) -> EventBus:
    """Return a Kafka-backed bus when configured, else a no-op bus.

    The selection rules are:

    * ``config.enabled`` is False                  → :class:`NullEventBus`
    * ``config.bootstrap_servers`` is empty        → :class:`NullEventBus`
    * ``kafka-python`` is not installed            → :class:`NullEventBus`
    * otherwise                                    → :class:`KafkaEventBus`

    Tests should construct ``InMemoryEventBus`` directly rather than going
    through this factory.
    """
    if config is None:
        config = EventBusConfig.from_env()

    if not config.enabled:
        logger.info("EventBus disabled via config; using NullEventBus.")
        return NullEventBus()

    if not config.bootstrap_servers:
        logger.info(
            "No Kafka brokers configured (CHIMERA_KAFKA_BROKERS); "
            "using NullEventBus."
        )
        return NullEventBus()

    try:
        from .kafka_bus import KafkaEventBus
    except ImportError as exc:
        logger.warning(
            "kafka-python not installed (%s); falling back to NullEventBus.",
            exc,
        )
        return NullEventBus()

    return KafkaEventBus(config)
