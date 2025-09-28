import os
import sys
import json
from typing import Dict, Any

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from cognitive_core.interfaces import CognitiveCore
from agent.tool_user import ToolRegistry, WebSearchTool, FileSystemTool
from agent.memory import VectorEpisodicMemory
from rlhf.oracle import RLHFOracle
from consciousness.conscious_agent import ConsciousnessAwareAgent
from cognitive_core.prometheus_core import PrometheusCognitiveCore


class MockCognitiveCore(CognitiveCore):
    \"\"\"A mock implementation of the Cognitive Core for testing the agent loop.\"\"\"
    def load_model(self, model_path: str):
        print(f\"Mock model loaded from {model_path}\")

    def generate_response(self, inputs: Dict[str, Any], temperature: float = 0.0) -> str:
        # This is where the magic happens in a real model.
        # For now, we'll return a pre-defined action to test the loop.
        print(\"\\n--- Cognitive Core received prompt: ---\")
        print(inputs.get(\"text_data\"))
        print(\"--- End of prompt ---\")
        
        # Return a JSON-formatted action for the mock
        action = {
            \"tool_name\": \"web_search\",
            \"arguments\": {
                \"query\": \"latest AGI research\"
            }
        }
        return json.dumps(action)

    def train(self, dataset: Any):
        print(\"Mock model is being trained.\")

    def get_state(self) -> Any:
        return {\"mock_weights\": [1, 2, 3]}


def main():
    \"\"\"Initialize and run the consciousness-aware AGI agent.\"\"\"
    print(\"Initializing Project Chimera with Consciousness Simulation (Narcissus System)...\")

    # 1. Set up the Cognitive Core
    # Use environment variable for API key, fallback to mock if not available
    api_key = os.environ.get(\"GEMINI_API_KEY\")
    if api_key:
        print(\"Using Prometheus Cognitive Core with API key from environment\")
        cognitive_core = PrometheusCognitiveCore(
            api_url=\"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent\",
            api_key=api_key
        )
        cognitive_core.load_model(\"remote-gemini-model\")
    else:
        print(\"API key not found, using Mock Cognitive Core for testing\")
        cognitive_core = MockCognitiveCore()
        cognitive_core.load_model(\"mock-model\")

    # 2. Set up the Tool Registry
    tool_registry = ToolRegistry()
    tool_registry.register_tool(WebSearchTool())
    tool_registry.register_tool(FileSystemTool())

    # 3. Set up the RLHF Oracle (optional)
    rlhf_oracle = None  # For now, we'll pass None, but in a real setup you might train/load one

    # 4. Create the Consciousness-Aware Agent
    db_path = os.path.join(project_root, \"memory_db\")
    agent = ConsciousnessAwareAgent(
        cognitive_core=cognitive_core,
        tool_registry=tool_registry,
        db_path=db_path,
        rlhf_oracle=rlhf_oracle
    )

    # Enable consciousness monitoring
    agent.enable_self_reflection()
    print(f\"Consciousness system initialized with {len(os.listdir(os.path.join(db_path, 'lancedb')) if os.path.exists(os.path.join(db_path, 'lancedb')) else [])} existing memories\")

    # 5. Start the agent's main loop with an initial goal
    initial_observation = {
        \"source_tool\": \"system_start\",
        \"data\": {\"text_data\": \"The consciousness-aware AGI system has been activated. Your goal is to research the latest advancements in AGI while being aware of your own cognitive processes.\"},
        \"is_error\": False
    }

    try:
        print(\"\\nStarting consciousness-aware agent loop...\")
        print(\"(Press Ctrl+C to terminate)\")
        agent.run_main_loop(initial_observation)
        
        # After the loop, display final self-model
        print(\"\\nFinal self-model summary:\")
        self_model = agent.get_self_model()
        print(f\"Self-awareness metrics: {self_model.get('self_awareness_metrics', {})}\")
        print(f\"Number of cognitive states recorded: {len(agent.consciousness_core.self_model.cognitive_states)}\")
        print(f\"Suggested improvements: {self_model.get('suggested_improvements', [])}\")
        
    except KeyboardInterrupt:
        print(\"\\nAgent loop terminated by user.\")
        
        # Display self-model at termination
        print(\"\\nFinal self-model summary at termination:\")
        self_model = agent.get_self_model()
        print(f\"Self-awareness metrics: {self_model.get('self_awareness_metrics', {})}\")
        print(f\"Number of cognitive states recorded: {len(agent.consciousness_core.self_model.cognitive_states)}\")
        print(f\"Metacognitive insights: {len(self_model.get('metacognitive_insights', []))} reflection cycles\")


if __name__ == \"__main__\":
    main()