from typing import Any

import pytest

from chimera.consciousness.integration import ConsciousnessIntegration
from chimera.consciousness.narcissus_core import NarcissusConsciousnessCore
from chimera.cognitive_core.interfaces import CognitiveCore


class FakeVectorEpisodicMemory:
    def __init__(self, db_path: str, table_name: str = "experiences"):
        self.db_path = db_path
        self.table_name = table_name
        self.records = []

    def remember(self, experience):
        self.records.append(experience)

    def recall(self, query: str, top_k: int = 5):
        return []


class TestCognitiveCore(CognitiveCore):
    """Test cognitive core for the consciousness system"""
    
    def load_model(self, model_path: str):
        print(f"Test cognitive core loaded from {model_path}")
    
    def generate_response(self, inputs: dict, temperature: float = 0.7) -> str:
        return "Test response for consciousness simulation"
    
    def train(self, dataset):
        print("Test cognitive core training")
    
    def get_state(self) -> dict:
        return {"test_weights": [1, 2, 3]}

@pytest.fixture(autouse=True)
def fake_memory(monkeypatch):
    monkeypatch.setattr(
        "chimera.consciousness.narcissus_core.VectorEpisodicMemory",
        FakeVectorEpisodicMemory,
    )


def test_consciousness_system(tmp_path):
    test_core = TestCognitiveCore()
    consciousness_core = NarcissusConsciousnessCore(
        cognitive_core=test_core,
        memory_db_path=str(tmp_path),
    )

    for i in range(3):
        consciousness_core.record_cognitive_state(
            thought_process=f"Thought process #{i+1}",
            attention_weights={"memory": 0.4, "reasoning": 0.3, "creativity": 0.3},
            decision_path=[f"step_{j}" for j in range(i+1)],
            confidence=0.7 + (i * 0.1),
            emotional_state={"curiosity": 0.8, "focus": 0.7},
            memory_context=[f"context_item_{k}" for k in range(3)],
            processing_load=0.3 + (i * 0.2)
        )
    analysis = consciousness_core.perform_introspective_analysis()
    assert analysis["self_model_snapshot"]["cognitive_states_count"] == 3
    assert "self_awareness_metrics" in analysis

    simulation = consciousness_core.simulate_self_modification(
        "Increase creativity module weight",
        {"module": "creativity", "change": "increase_weight", "value": 0.1}
    )
    assert simulation["proposal"] == "Increase creativity module weight"


def test_consciousness_with_mock_agent(tmp_path):
    test_core = TestCognitiveCore()
    consciousness_core = NarcissusConsciousnessCore(
        cognitive_core=test_core,
        memory_db_path=str(tmp_path),
    )
    integration = ConsciousnessIntegration(consciousness_core)

    for cycle in range(5):
        integration.record_cognitive_state_from_agent(
            thought_process=f"Processing observation #{cycle}",
            attention_weights={
                "environment": 0.4,
                "memory": 0.3, 
                "planning": 0.2,
                "self_reflection": 0.1
            },
            decision_path=["observe", "analyze", "decide", "act"],
            confidence=0.6 + (cycle * 0.1),
            memory_context=["recent_event_1", "recent_event_2"],
            processing_load=0.5
        )

    insights = integration.get_consciousness_insights()
    assert insights["self_model_snapshot"]["cognitive_states_count"] == 5

    simulation = integration.simulate_cognitive_change(
        "Optimize decision-making process",
        {"process": "decision_making", "optimization": "parallel_evaluation"}
    )
    assert "risk_assessment" in simulation
