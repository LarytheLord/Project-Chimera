import json
from typing import Any, Dict

import pytest

from chimera.agent.agent import Agent
from chimera.agent.tool_user import Tool, ToolRegistry
from chimera.cognitive_core.interfaces import CognitiveCore
from chimera.main import (
    build_initial_observation,
    create_tool_registry,
    main,
    run_agent_turn,
)


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


class MockCognitiveCore(CognitiveCore):
    def __init__(self, action_json: Dict[str, Any]):
        self.action_json = action_json

    def generate_response(self, prompt: Dict[str, Any]) -> str:
        return json.dumps(self.action_json)

    def load_model(self, model_path: str):
        pass

    def train(self, dataset: Any):
        pass

    def get_state(self) -> Any:
        return None


class SumTool(Tool):
    @property
    def name(self) -> str:
        return "sum_numbers"

    @property
    def description(self) -> str:
        return "Adds two numbers."

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        }

    def __call__(self, a: int, b: int) -> Any:
        return a + b


@pytest.fixture(autouse=True)
def fake_embedding_model(monkeypatch):
    fake_lancedb = FakeLanceDB()
    monkeypatch.setattr(
        "chimera.agent.memory._load_vector_dependencies",
        lambda: (fake_lancedb, FakeArrow, FakeSentenceTransformer),
    )


def test_build_initial_observation():
    observation = build_initial_observation("hello there")
    assert observation == {"source": "user", "data": {"text_data": "hello there"}}


def test_create_tool_registry_defaults_to_web_only():
    registry = create_tool_registry()
    assert "web_search" in registry.get_tool_names()
    assert "file_system" not in registry.get_tool_names()


def test_create_tool_registry_can_enable_file_system():
    registry = create_tool_registry(allow_file_system=True)
    assert "file_system" in registry.get_tool_names()


def test_run_agent_turn_records_memory(tmp_path):
    registry = ToolRegistry()
    registry.register_tool(SumTool())
    agent = Agent(
        cognitive_core=MockCognitiveCore(
            {"tool_name": "sum_numbers", "arguments": {"a": 2, "b": 3}}
        ),
        tool_registry=registry,
        db_path=str(tmp_path),
    )

    turn = run_agent_turn(agent, "add these numbers")

    assert turn["action"]["tool_name"] == "sum_numbers"
    assert turn["outcome"]["data"]["text_data"] == 5
    assert len(agent.working_memory.get_context()) == 3
    assert agent.episodic_memory.table.count_rows() == 1


class FakeWorkingMemory:
    def __init__(self):
        self.history = []

    def add(self, record):
        self.history.append(record)


class FakeEpisodicMemory:
    def __init__(self):
        self.remembered = []

    def remember(self, experience):
        self.remembered.append(experience)


class FakeAgent:
    def __init__(self):
        self.working_memory = FakeWorkingMemory()
        self.episodic_memory = FakeEpisodicMemory()

    def _think(self, observation):
        return {"tool_name": "demo_tool", "arguments": {"topic": observation["data"]["text_data"]}}

    def _act(self, action):
        return {"source_tool": action["tool_name"], "data": {"text_data": "ok"}, "is_error": False}


def test_main_single_prompt_json_output(monkeypatch, capsys):
    monkeypatch.setattr("chimera.main.create_agent", lambda **_kwargs: FakeAgent())

    exit_code = main(["--prompt", "demo prompt", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["action"]["tool_name"] == "demo_tool"
    assert payload["outcome"]["source_tool"] == "demo_tool"
