from typing import Any

import pytest

from src import NarcissusConsciousnessCore
from src import CognitiveCore


class FakeVectorEpisodicMemory:
    def __init__(self, db_path: str, table_name: str = "experiences"):
        self.db_path = db_path
        self.table_name = table_name

    def remember(self, experience):
        return None

    def recall(self, query: str, top_k: int = 5):
        return []


class TestCognitiveCore(CognitiveCore):
    def load_model(self, model_path: str):
        return None

    def generate_response(self, inputs: dict, temperature: float = 0.7) -> str:
        return "Test response"

    def train(self, dataset: Any):
        return None

    def get_state(self) -> dict:
        return {"status": "ok"}


@pytest.fixture(autouse=True)
def fake_memory(monkeypatch):
    monkeypatch.setattr(
        "chimera.consciousness.narcissus_core.VectorEpisodicMemory",
        FakeVectorEpisodicMemory,
    )


def test_introspective_analysis_shape(tmp_path):
    consciousness_core = NarcissusConsciousnessCore(
        cognitive_core=TestCognitiveCore(),
        memory_db_path=str(tmp_path),
    )
    consciousness_core.record_cognitive_state(
        thought_process="Testing simple introspection",
        attention_weights={"memory": 0.5, "planning": 0.5},
        decision_path=["observe", "reflect"],
        confidence=0.75,
        emotional_state={"curiosity": 0.8},
        memory_context=["hello"],
        processing_load=0.2,
    )

    analysis = consciousness_core.perform_introspective_analysis()

    assert set(analysis) == {
        "self_model_snapshot",
        "metacognitive_insights",
        "suggested_improvements",
        "self_awareness_metrics",
    }
