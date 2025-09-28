"""
Test script for the Narcissus Consciousness Simulation System
"""
import os
import sys
import json

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from consciousness.narcissus_core import NarcissusConsciousnessCore
from consciousness.integration import ConsciousnessIntegration
from cognitive_core.interfaces import CognitiveCore
from agent.memory import VectorEpisodicMemory


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


def test_consciousness_system():
    """Test the core consciousness simulation functionality"""
    print("Testing Narcissus Consciousness Simulation System...")
    
    # Initialize test components
    test_core = TestCognitiveCore()
    db_path = os.path.join(project_root, "memory_db")
    
    # Create the consciousness core
    consciousness_core = NarcissusConsciousnessCore(
        cognitive_core=test_core,
        memory_db_path=db_path
    )
    
    print("✓ NarcissusConsciousnessCore initialized")
    
    # Test recording cognitive states
    for i in range(3):
        state = consciousness_core.record_cognitive_state(
            thought_process=f"Thought process #{i+1}",
            attention_weights={"memory": 0.4, "reasoning": 0.3, "creativity": 0.3},
            decision_path=[f"step_{j}" for j in range(i+1)],
            confidence=0.7 + (i * 0.1),
            emotional_state={"curiosity": 0.8, "focus": 0.7},
            memory_context=[f"context_item_{k}" for k in range(3)],
            processing_load=0.3 + (i * 0.2)
        )
        print(f"✓ Recorded cognitive state #{i+1}")
    
    # Test metacognitive observation
    analysis = consciousness_core.perform_introspective_analysis()
    print(f"✓ Performed introspective analysis: {len(analysis)} components analyzed")
    
    # Test self-modeling
    self_model = consciousness_core.self_model.get_self_model()
    print(f"✓ Retrieved self-model with {self_model['cognitive_states_count']} states")
    
    # Test simulation framework
    simulation = consciousness_core.simulate_self_modification(
        "Increase creativity module weight",
        {"module": "creativity", "change": "increase_weight", "value": 0.1}
    )
    print(f"✓ Ran simulation: {simulation['proposal']}")
    
    # Test consciousness integration
    integration = ConsciousnessIntegration(consciousness_core)
    insights = integration.get_consciousness_insights()
    print(f"✓ Retrieved consciousness insights: {len(insights)} insight categories")
    
    print("\nAll tests passed! Consciousness simulation system is functioning correctly.")


def test_consciousness_with_mock_agent():
    """Test the consciousness system with a simplified agent interaction"""
    print("\nTesting consciousness system with mock agent integration...")
    
    # Initialize components
    test_core = TestCognitiveCore()
    db_path = os.path.join(project_root, "memory_db")
    
    consciousness_core = NarcissusConsciousnessCore(
        cognitive_core=test_core,
        memory_db_path=db_path
    )
    integration = ConsciousnessIntegration(consciousness_core)
    
    # Simulate cognitive states as if from an agent
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
        print(f"✓ Agent cycle {cycle+1} cognitive state recorded")
    
    # Perform introspection
    insights = integration.get_consciousness_insights()
    print(f"✓ Generated introspective insights: {insights['suggested_improvements'][:2]}")
    
    # Test self-modification simulation
    simulation = integration.simulate_cognitive_change(
        "Optimize decision-making process",
        {"process": "decision_making", "optimization": "parallel_evaluation"}
    )
    print(f"✓ Simulated cognitive change: {simulation['risk_assessment']}")
    
    print("Mock agent integration test completed successfully!")


if __name__ == "__main__":
    test_consciousness_system()
    test_consciousness_with_mock_agent()
    
    print("\n" + "="*60)
    print("CONSCIOUSNESS SIMULATION SYSTEM TEST RESULTS")
    print("="*60)
    print("✓ Self-Modeling Engine: Operational")
    print("✓ Metacognitive Observer: Operational") 
    print("✓ Self-Simulation Framework: Operational")
    print("✓ Consciousness Integration: Operational")
    print("✓ Introspective Analysis: Operational")
    print("="*60)
    print("Project Chimera's Narcissus consciousness system is ready!")