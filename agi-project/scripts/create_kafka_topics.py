"""Pre-create the Chimera topics with sane retention defaults.

This is optional — the agent will auto-create topics on first publish — but
running this once gives you predictable retention and partition counts.

Usage:
    export CHIMERA_KAFKA_BROKERS=localhost:9092
    python scripts/create_kafka_topics.py
"""

from __future__ import annotations

import os
import sys

from chimera.eventbus import ALL_TOPICS, EventBusConfig


# 7 days of replayable agent traces is plenty for research / debugging.
DEFAULT_RETENTION_MS = 7 * 24 * 60 * 60 * 1000
DEFAULT_PARTITIONS = 3
DEFAULT_REPLICATION = 1  # single-node dev cluster


def main() -> int:
    cfg = EventBusConfig.from_env()
    if not cfg.bootstrap_servers:
        print(
            "CHIMERA_KAFKA_BROKERS is not set. Pass it as an env var, e.g.\n"
            "  CHIMERA_KAFKA_BROKERS=localhost:9092 python scripts/create_kafka_topics.py"
        )
        return 1

    try:
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka.errors import TopicAlreadyExistsError
    except ImportError:
        print("kafka-python is required. Install with: pip install kafka-python")
        return 1

    admin = KafkaAdminClient(
        bootstrap_servers=cfg.bootstrap_servers,
        client_id="chimera-topic-bootstrap",
    )

    new_topics = [
        NewTopic(
            name=cfg.topic(t),
            num_partitions=DEFAULT_PARTITIONS,
            replication_factor=DEFAULT_REPLICATION,
            topic_configs={"retention.ms": str(DEFAULT_RETENTION_MS)},
        )
        for t in ALL_TOPICS
    ]

    for topic in new_topics:
        try:
            admin.create_topics([topic])
            print(f"created topic: {topic.name}")
        except TopicAlreadyExistsError:
            print(f"topic exists:  {topic.name}")

    admin.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
