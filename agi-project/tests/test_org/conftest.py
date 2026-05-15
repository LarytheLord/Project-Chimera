"""Test fixtures for the org module.

The fake_vector_deps fixture patches `chimera.agent.memory._load_vector_dependencies`
so VectorEpisodicMemory can be instantiated without lancedb/sentence-transformers.

RoleAwareMockCognitiveCore routes responses by the [[ROLE:Name]] marker that
chimera.org.prompts injects into every role prompt -- so a single mock can serve
all 6 roles in a sequential run.
"""

from __future__ import annotations

import json
import re
import sys
import types
from typing import Any

# chimera/__init__.py eagerly re-exports modules that import torch / transformers / trl /
# httpx at module load time. For these unit tests we don't exercise those paths, so we
# install minimal stub modules in sys.modules BEFORE importing anything from chimera.
def _install_stub(name: str) -> None:
    if name in sys.modules:
        return
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            mod = types.ModuleType(sub)
            mod.__path__ = []  # mark as package so submodule imports work
            sys.modules[sub] = mod


for _name in (
    "httpx",
    "torch",
    "transformers",
    "trl",
    "datasets",
    "ddgs",
    "bs4",
    "requests",
):
    _install_stub(_name)

# Provide the specific attributes that chimera modules import by name.
sys.modules["transformers"].pipeline = lambda *a, **k: None
sys.modules["transformers"].AutoTokenizer = type("AutoTokenizer", (), {})
sys.modules["transformers"].AutoModelForSequenceClassification = type(
    "AutoModelForSequenceClassification", (), {}
)
sys.modules["trl"].RewardTrainer = type("RewardTrainer", (), {})
sys.modules["trl"].RewardConfig = type("RewardConfig", (), {})
sys.modules["datasets"].Dataset = type("Dataset", (), {})

import pytest

from chimera.cognitive_core.interfaces import CognitiveCore


# ---------- Fakes for vector memory ----------

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


# ---------- Role-aware mock cognitive core ----------

class RoleAwareMockCognitiveCore(CognitiveCore):
    """Returns canned JSON responses keyed by [[ROLE:Name]] marker in the prompt.

    Pass `responses={"CEO": {"output": {...}, "next_role": "RnD"}, ...}` to wire each
    role independently. Call `set_response("CEO", new_response)` mid-test to change
    behavior dynamically (e.g. for the QA-reject-and-retry test).
    """

    def __init__(self, responses: dict[str, dict] | None = None):
        self.responses: dict[str, dict] = dict(responses or {})
        self.call_history: list[tuple[str, str]] = []  # (role_name, prompt)

    def set_response(self, role_name: str, response: dict) -> None:
        self.responses[role_name] = response

    def generate_response(self, inputs: dict, **kwargs) -> str:
        prompt = inputs.get("text_data", "") if isinstance(inputs, dict) else str(inputs)
        match = re.search(r"\[\[ROLE:([A-Za-z]+)\]\]", prompt)
        role = match.group(1) if match else "UNKNOWN"
        self.call_history.append((role, prompt))
        response = self.responses.get(role)
        if response is None:
            response = {"output": {"error": f"no canned response for {role}"}, "next_role": None}
        return json.dumps(response)

    def load_model(self, model_path: str) -> None:
        pass

    def train(self, dataset: Any) -> None:
        pass

    def get_state(self) -> Any:
        return None


# ---------- Default happy-path responses ----------

def make_happy_path_responses() -> dict[str, dict]:
    return {
        "CEO": {
            "output": {
                "brief": "Draft an announcement for the new vector embeddings feature.",
                "success_criteria": [
                    "Clear value proposition",
                    "Three paragraphs",
                    "Target developer audience",
                ],
                "risks": ["Audience may need more technical depth"],
            },
            "next_role": "RnD",
        },
        "RnD": {
            "output": {
                "research_summary": "Vector embeddings let semantic search work on free text.",
                "approach": "Lead with a concrete use case, then describe the technology.",
                "open_questions": [],
            },
            "next_role": "Marketing",
        },
        "Marketing": {
            "output": {
                "positioning": "Search your codebase by meaning, not keywords.",
                "key_messages": [
                    "Drop-in replacement for keyword search",
                    "Free to run locally",
                ],
                "audience": "Developers",
            },
            "next_role": "Production",
        },
        "Production": {
            "output": {
                "deliverable": "We're excited to launch vector embeddings...",
            },
            "next_role": "Ops",
        },
        "Ops": {
            "output": {
                "channel": "email",
                "recipient": "users@example.com",
                "subject": "New: vector embeddings in Chimera",
                "body_reference": "deliverable",
                "when": "now",
            },
            "next_role": "QA",
        },
        "QA": {
            "output": {
                "verdict": "approved",
                "reason": "Meets all three success criteria; Ops plan is safe.",
            },
            "next_role": None,
        },
    }


# ---------- Pytest fixtures ----------

@pytest.fixture(autouse=True)
def fake_vector_deps(monkeypatch):
    fake_lance = FakeLanceDB()
    monkeypatch.setattr(
        "chimera.agent.memory._load_vector_dependencies",
        lambda: (fake_lance, FakeArrow, FakeSentenceTransformer),
    )
    return fake_lance


@pytest.fixture
def tmp_db_root(tmp_path):
    root = tmp_path / "chimera_org"
    root.mkdir()
    return str(root)


@pytest.fixture
def happy_responses():
    return make_happy_path_responses()


@pytest.fixture
def happy_core(happy_responses):
    return RoleAwareMockCognitiveCore(happy_responses)
