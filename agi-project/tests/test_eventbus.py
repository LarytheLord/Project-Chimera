"""Tests for the Chimera event bus and its integration into the agent loop."""

import json
from typing import Any, Dict

import pytest

from chimera.eventbus import (
    Event,
    EventBusConfig,
    InMemoryEventBus,
    NullEventBus,
    Topics,
    build_event_bus,
)
from chimera.eventbus.events import new_event
from chimera.eventbus.replay import replay_from_iterable


# --- Helpers reused by the agent integration test below -----------------

class FakeSentenceTransformer:
    def __init__(self, *_args, **_kwargs):
        pass

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, text: str):
        length = float(len(text))
        return [length, length / 2.0, 1.0]


class FakeArrow:
    @staticmethod
    def schema(fields):
        return fields

    @staticmethod
    def field(name, data_type):
        return (name, data_type)

    @staticmethod
    def list_(data_type, embedding_dim):
        return (data_type, embedding_dim)

    @staticmethod
    def float32():
        return "float32"

    @staticmethod
    def string():
        return "string"


class FakeResults:
    def __init__(self, rows):
        self.rows = rows

    def iterrows(self):
        for index, row in enumerate(self.rows):
            yield index, row


class FakeSearch:
    def __init__(self, rows):
        self.rows = rows
        self._limit = len(rows)

    def limit(self, top_k: int):
        self._limit = top_k
        return self

    def to_df(self):
        return FakeResults(self.rows[: self._limit])


class FakeTable:
    def __init__(self):
        self.rows = []

    def add(self, rows):
        self.rows.extend(rows)

    def count_rows(self):
        return len(self.rows)

    def search(self, query_vector):
        return FakeSearch(self.rows)


class FakeDB:
    def __init__(self):
        self.tables = {}

    def table_names(self):
        return list(self.tables.keys())

    def open_table(self, table_name):
        return self.tables[table_name]

    def create_table(self, table_name, schema):
        table = FakeTable()
        self.tables[table_name] = table
        return table


class FakeLanceDB:
    def __init__(self):
        self.databases = {}

    def connect(self, path):
        if path not in self.databases:
            self.databases[path] = FakeDB()
        return self.databases[path]


# --- Event / serialization tests ----------------------------------------

def test_event_round_trip():
    event = new_event(
        topic=Topics.PERCEPTION,
        payload={"observation": {"task": "hello"}},
        trace_id="trace-1",
        session_id="sess-1",
        agent_id="agent-1",
        event_type="perception",
    )
    raw = event.to_json()
    restored = Event.from_json(raw)

    assert restored.topic == event.topic
    assert restored.payload == event.payload
    assert restored.trace_id == "trace-1"
    assert restored.session_id == "sess-1"
    assert restored.agent_id == "agent-1"
    assert restored.event_type == "perception"
    assert restored.schema_version == event.schema_version


def test_event_serializes_non_json_payload():
    """Payloads with awkward objects should still round-trip via str fallback."""
    class Weird:
        def __repr__(self):
            return "<weird>"

    event = new_event(topic=Topics.ACTIONS, payload={"obj": Weird()})
    raw = event.to_json()
    parsed = json.loads(raw)
    assert parsed["payload"]["obj"] == "<weird>"


# --- Bus implementations ------------------------------------------------

def test_null_bus_returns_event_but_drops_it():
    bus = NullEventBus()
    event = bus.publish(Topics.ACTIONS, {"a": 1}, trace_id="t")
    assert event.topic == Topics.ACTIONS
    # Null bus has no history / no errors on flush+close
    bus.flush(timeout=0.1)
    bus.close()


def test_inmemory_bus_records_events_and_notifies_subscribers():
    bus = InMemoryEventBus()
    received = []
    bus.subscribe(Topics.PERCEPTION, received.append)

    bus.publish(Topics.PERCEPTION, {"observation": "x"}, trace_id="t1")
    bus.publish(Topics.PERCEPTION, {"observation": "y"}, trace_id="t2")

    assert len(received) == 2
    assert received[0].payload["observation"] == "x"
    assert received[1].trace_id == "t2"

    history = bus.history(Topics.PERCEPTION)
    assert [e.payload["observation"] for e in history] == ["x", "y"]


def test_inmemory_bus_drain_clears_history():
    bus = InMemoryEventBus()
    bus.publish(Topics.AGENT_TRACES, {"phase": "loop_start"}, trace_id="t")
    drained = bus.drain(Topics.AGENT_TRACES)
    assert len(drained) == 1
    assert bus.history(Topics.AGENT_TRACES) == []


# --- Factory + config ---------------------------------------------------

def test_build_event_bus_returns_null_when_no_brokers(monkeypatch):
    cfg = EventBusConfig.from_env(env={})
    bus = build_event_bus(cfg)
    assert isinstance(bus, NullEventBus)


def test_build_event_bus_returns_null_when_disabled():
    cfg = EventBusConfig(bootstrap_servers=["localhost:9092"], enabled=False)
    bus = build_event_bus(cfg)
    assert isinstance(bus, NullEventBus)


def test_event_bus_config_from_env_parses_brokers():
    env = {
        "CHIMERA_KAFKA_BROKERS": "broker1:9092,broker2:9092",
        "CHIMERA_AGENT_ID": "agent-7",
    }
    cfg = EventBusConfig.from_env(env=env)
    assert cfg.bootstrap_servers == ["broker1:9092", "broker2:9092"]
    assert cfg.agent_id == "agent-7"
    assert cfg.enabled is True


def test_topic_prefix_is_applied():
    cfg = EventBusConfig(topic_prefix="dev.")
    assert cfg.topic("chimera.perception") == "dev.chimera.perception"


# --- Replay helper ------------------------------------------------------

def test_replay_groups_events_by_trace():
    e1 = new_event(Topics.PERCEPTION, {"observation": "a"}, trace_id="t1")
    e2 = new_event(Topics.ACTIONS, {"action": {}}, trace_id="t1")
    e3 = new_event(Topics.PERCEPTION, {"observation": "b"}, trace_id="t2")
    e4 = new_event(Topics.ACTIONS, {"action": {}})  # no trace_id

    groups = replay_from_iterable([e1, e2, e3, e4])

    assert set(groups) == {"t1", "t2", "_untraced"}
    assert [e.topic for e in groups["t1"]] == [Topics.PERCEPTION, Topics.ACTIONS]
    assert len(groups["t2"]) == 1
    assert len(groups["_untraced"]) == 1


# --- KafkaEventBus configuration guard ----------------------------------

def test_kafka_bus_requires_brokers():
    from chimera.eventbus.kafka_bus import KafkaEventBus

    with pytest.raises(ValueError):
        KafkaEventBus(EventBusConfig(bootstrap_servers=[]))


# --- Agent loop integration --------------------------------------------

@pytest.fixture(autouse=True)
def fake_embedding_model(monkeypatch):
    fake_lancedb = FakeLanceDB()
    monkeypatch.setattr(
        "chimera.agent.memory._load_vector_dependencies",
        lambda: (fake_lancedb, FakeArrow, FakeSentenceTransformer),
    )


@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path)


def _make_agent(db_path: str, bus: InMemoryEventBus):
    from chimera import Agent, CognitiveCore, Tool, ToolRegistry

    class ExitTool(Tool):
        @property
        def name(self) -> str:
            return "exit"

        @property
        def description(self) -> str:
            return "Stops the agent loop."

        def get_schema(self) -> Dict[str, Any]:
            return {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            }

        def __call__(self, **_kwargs) -> str:
            return "stopping"

    class OneShotCore(CognitiveCore):
        """Returns a single action that triggers the loop's exit branch."""

        def load_model(self, model_path: str) -> None:
            pass

        def generate_response(self, inputs, temperature: float = 0.7) -> str:
            return json.dumps({"tool_name": "exit", "arguments": {}})

        def train(self, dataset) -> None:
            pass

        def get_state(self):
            return {}

    registry = ToolRegistry()
    registry.register_tool(ExitTool())
    return Agent(
        cognitive_core=OneShotCore(),
        tool_registry=registry,
        db_path=db_path,
        event_bus=bus,
        agent_id="test-agent",
        session_id="test-session",
    )


def test_agent_loop_emits_full_event_sequence(temp_db_path):
    bus = InMemoryEventBus()
    agent = _make_agent(temp_db_path, bus)

    agent.run_main_loop({"task": "demo"})

    perceptions = bus.history(Topics.PERCEPTION)
    actions = bus.history(Topics.ACTIONS)
    tool_calls = bus.history(Topics.TOOL_CALLS)
    memory_writes = bus.history(Topics.MEMORY_WRITES)
    traces = bus.history(Topics.AGENT_TRACES)

    # Exactly one cycle (the exit tool short-circuits the loop).
    assert len(perceptions) == 1
    assert len(actions) == 1
    # tool_calls topic gets both invoke + result events.
    assert len(tool_calls) == 2
    assert {e.event_type for e in tool_calls} == {"tool_invoke", "tool_result"}
    assert len(memory_writes) == 1
    # AGENT_TRACES contains loop_start, cycle_complete, loop_end.
    trace_phases = [e.payload.get("phase") for e in traces]
    assert "loop_start" in trace_phases
    assert "cycle_complete" in trace_phases
    assert "loop_end" in trace_phases

    # Per-cycle events all share a single trace_id...
    cycle_trace_id = perceptions[0].trace_id
    assert cycle_trace_id is not None
    for event in (*actions, *tool_calls, *memory_writes):
        assert event.trace_id == cycle_trace_id
    # ...and all events carry the agent + session identity.
    for event in (*perceptions, *actions, *tool_calls, *memory_writes, *traces):
        assert event.agent_id == "test-agent"
        assert event.session_id == "test-session"


def test_agent_works_with_null_bus_default(temp_db_path):
    """Agent must run even when no bus is provided (NullEventBus default)."""
    from chimera import Agent, CognitiveCore, Tool, ToolRegistry

    class ExitTool(Tool):
        @property
        def name(self) -> str:
            return "exit"

        @property
        def description(self) -> str:
            return "Stops the agent loop."

        def get_schema(self) -> Dict[str, Any]:
            return {"name": self.name, "description": self.description,
                    "parameters": {"type": "object", "properties": {}}}

        def __call__(self, **_kwargs) -> str:
            return "stopping"

    class OneShotCore(CognitiveCore):
        def load_model(self, model_path: str) -> None: pass

        def generate_response(self, inputs, temperature: float = 0.7) -> str:
            return json.dumps({"tool_name": "exit", "arguments": {}})

        def train(self, dataset) -> None: pass

        def get_state(self): return {}

    registry = ToolRegistry()
    registry.register_tool(ExitTool())
    agent = Agent(
        cognitive_core=OneShotCore(),
        tool_registry=registry,
        db_path=temp_db_path,
    )

    # Should complete without raising.
    agent.run_main_loop({"task": "demo"})
    assert isinstance(agent.event_bus, NullEventBus)
