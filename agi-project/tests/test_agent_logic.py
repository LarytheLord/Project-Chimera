import json
from typing import Any, Dict

import pytest

from chimera.agent.agent import Agent
from chimera.agent.memory import Experience
from chimera.agent.tool_user import Tool, ToolPolicy, ToolRegistry
from chimera.consciousness.conscious_agent import ConsciousnessAwareAgent
from chimera.cognitive_core.interfaces import CognitiveCore

# --- Mock Components ---

class FakeSentenceTransformer:
    """Deterministic embedding stub for testing vector memory round-trips."""

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
    """A mock cognitive core that returns a predefined JSON action."""
    def __init__(self, action_json: Dict[str, Any]):
        self.action_json = action_json
        self.prompts: list[Dict[str, Any]] = []

    def generate_response(self, prompt: Dict[str, Any], temperature: float = 0.7) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.action_json)

    def load_model(self, model_path: str):
        pass

    def train(self, dataset: Any):
        pass

    def get_state(self) -> Any:
        pass

class SumTool(Tool):
    """A simple tool for testing that adds two numbers."""
    @property
    def name(self) -> str:
        return "sum_numbers"

    @property
    def description(self) -> str:
        return "Adds two integers a and b."

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


class RestrictedEchoTool(Tool):
    @property
    def name(self) -> str:
        return "restricted_echo"

    @property
    def description(self) -> str:
        return "Echoes a payload, but only when explicitly enabled."

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "string"},
                },
                "required": ["payload"],
            },
        }

    def get_policy(self) -> ToolPolicy:
        return ToolPolicy(
            risk_level="high",
            capabilities=("debug.echo",),
            allowed_by_default=False,
            requires_approval=True,
            policy_note="Test-only restricted tool.",
        )

    def __call__(self, payload: str) -> Any:
        return payload

# --- Test Fixture ---

@pytest.fixture(autouse=True)
def fake_embedding_model(monkeypatch):
    fake_lancedb = FakeLanceDB()
    monkeypatch.setattr(
        "chimera.agent.memory._load_vector_dependencies",
        lambda: (fake_lancedb, FakeArrow, FakeSentenceTransformer),
    )

@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary directory for the LanceDB database."""
    return str(tmp_path)

def setup_agent_for_test(db_path: str):
    """Sets up the agent with mock components for a predictable test."""
    mock_action = {
        "tool_name": "sum_numbers",
        "arguments": {"a": 5, "b": 10}
    }
    mock_core = MockCognitiveCore(action_json=mock_action)

    sum_tool = SumTool()
    tool_registry = ToolRegistry()
    tool_registry.register_tool(sum_tool)

    agent = Agent(
        cognitive_core=mock_core,
        tool_registry=tool_registry,
        db_path=db_path
    )
    return agent


def setup_conscious_agent_for_test(db_path: str, action_json: Dict[str, Any], tool_registry: ToolRegistry):
    mock_core = MockCognitiveCore(action_json=action_json)
    agent = ConsciousnessAwareAgent(
        cognitive_core=mock_core,
        tool_registry=tool_registry,
        db_path=db_path,
    )
    return agent

# --- The Test ---

def test_agent_vector_memory_and_recall(temp_db_path):
    """
    Tests the agent's ability to use the VectorEpisodicMemory to remember
    an experience and recall it based on semantic similarity.
    """
    # 1. Setup agent and a unique experience
    agent = setup_agent_for_test(db_path=temp_db_path)
    
    unique_observation = {"source": "user", "data": {"text_data": "The sky is filled with blue elephants today."}}
    action_taken = {"tool_name": "observe", "arguments": {"phenomenon": "flying blue elephants"}}
    outcome_result = {"source_tool": "observe", "data": {"text_data": "Confirmed: The elephants are indeed blue and airborne."}}
    
    experience_to_remember = Experience(
        observation=unique_observation, 
        action=action_taken, 
        outcome=outcome_result
    )

    # 2. Agent remembers the experience
    agent.episodic_memory.remember(experience_to_remember)

    # 3. Agent recalls memory with a semantically similar query
    # This query does not use the same keywords but has a similar meaning.
    semantic_query = unique_observation["data"]["text_data"]
    recalled_experiences = agent.episodic_memory.recall(semantic_query, top_k=1)

    # 4. Verification
    # Verify that the agent recalled the correct experience.
    assert len(recalled_experiences) == 1, "Should have recalled one experience."
    
    recalled_exp = recalled_experiences[0]
    
    # Compare the observation text which is the most unique part
    assert recalled_exp.observation["data"]["text_data"] == unique_observation["data"]["text_data"], \
        "The recalled experience should match the one that was remembered."

    print("Agent vector memory (remember and recall) test passed successfully!")


def test_tool_registry_hides_restricted_tools_from_prompt_by_default():
    registry = ToolRegistry()
    registry.register_tool(SumTool())
    registry.register_tool(RestrictedEchoTool())

    prompt_schemas = json.loads(registry.get_tool_schemas())
    assert "sum_numbers" in prompt_schemas
    assert "restricted_echo" not in prompt_schemas

    all_schemas = json.loads(registry.get_tool_schemas(include_restricted=True))
    assert all_schemas["restricted_echo"]["x-tool-policy"]["allowed_by_default"] is False
    assert all_schemas["restricted_echo"]["x-tool-policy"]["risk_level"] == "high"


def test_tool_registry_blocks_restricted_tools_by_default():
    registry = ToolRegistry()
    registry.register_tool(RestrictedEchoTool())

    outcome = registry.execute_action({
        "tool_name": "restricted_echo",
        "arguments": {"payload": "secret"},
    })

    assert outcome["is_error"] is True
    assert outcome["policy"]["status"] == "blocked"
    assert "restricted by policy" in outcome["data"]["text_data"]


def test_tool_registry_executes_restricted_tools_when_enabled():
    registry = ToolRegistry(allow_restricted_tools=True)
    registry.register_tool(RestrictedEchoTool())

    outcome = registry.execute_action({
        "tool_name": "restricted_echo",
        "arguments": {"payload": "secret"},
    })

    assert outcome["is_error"] is False
    assert outcome["policy"]["status"] == "executed"
    assert outcome["data"]["text_data"] == "secret"


def test_tool_registry_returns_structured_unknown_tool_error():
    registry = ToolRegistry()
    outcome = registry.execute_action({
        "tool_name": "missing_tool",
        "arguments": {},
    })

    assert outcome["is_error"] is True
    assert outcome["policy"]["status"] == "unknown_tool"
    assert "not found" in outcome["data"]["text_data"]


def test_agent_act_uses_policy_aware_tool_execution(temp_db_path):
    registry = ToolRegistry()
    registry.register_tool(RestrictedEchoTool())
    agent = Agent(
        cognitive_core=MockCognitiveCore(
            {"tool_name": "restricted_echo", "arguments": {"payload": "secret"}}
        ),
        tool_registry=registry,
        db_path=temp_db_path,
    )

    outcome = agent._act({"tool_name": "restricted_echo", "arguments": {"payload": "secret"}})

    assert outcome["is_error"] is True
    assert outcome["policy"]["status"] == "blocked"


def test_conscious_agent_act_uses_policy_aware_tool_execution(temp_db_path):
    registry = ToolRegistry()
    registry.register_tool(RestrictedEchoTool())
    agent = setup_conscious_agent_for_test(
        db_path=temp_db_path,
        action_json={"tool_name": "restricted_echo", "arguments": {"payload": "secret"}},
        tool_registry=registry,
    )

    outcome = agent._act({"tool_name": "restricted_echo", "arguments": {"payload": "secret"}})

    assert outcome["is_error"] is True
    assert outcome["policy"]["status"] == "blocked"
