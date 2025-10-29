import os
import sys
import json
from typing import Dict, Any

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from cognitive_core.interfaces import CognitiveCore
from agent.tool_user import ToolRegistry, WebSearchTool, FileSystemTool
from agent.agent import Agent
from consciousness.conscious_agent import ConsciousnessAwareAgent

class MockCognitiveCore(CognitiveCore):
    """A mock implementation of the Cognitive Core for testing the agent loop."""
    def load_model(self, model_path: str):
        print(f"Mock model loaded from {model_path}")

    def generate_response(self, inputs: Dict[str, Any], temperature: float = 0.0) -> str:
        # This is where the magic happens in a real model.
        # For now, we'll return a pre-defined action to test the loop.
        print("\n--- Cognitive Core received prompt: ---")
        print(inputs.get("text_data"))
        print("--- End of prompt ---")
        action = {
            "tool_name": "file_system",
            "arguments": {
                "operation": "list_directory",
                "path": "src"
            }
        }
        return json.dumps(action)

    def train(self, dataset: Any):
        print("Mock model is being trained.")

    def get_state(self) -> Any:
        return {"mock_weights": [1, 2, 3]}

# --- Main Application ---

def main():
    """Initializes and runs the AGI agent with optional consciousness simulation."""
    print("Initializing Project Chimera...")
    
    # Option to enable consciousness simulation
    use_consciousness = os.environ.get("USE_CONSCIOUSNESS", "false").lower() == "true"

    # 1. Set up the Cognitive Core
    cognitive_core = MockCognitiveCore()
    cognitive_core.load_model("mock-model")

    # 2. Set up the Tool Registry
    tool_registry = ToolRegistry()
    tool_registry.register_tool(WebSearchTool())
    tool_registry.register_tool(FileSystemTool())

    # 3. Set up the RLHF Oracle
    # We don't need the oracle for this test, so we'll pass None.
    rlhf_oracle = None

    # 4. Create the appropriate Agent
    db_path = os.path.join(project_root, "memory_db")
    
    if use_consciousness:
        print("Using consciousness-aware agent...")
        agent = ConsciousnessAwareAgent(
            cognitive_core=cognitive_core,
            tool_registry=tool_registry,
            db_path=db_path,
            rlhf_oracle=rlhf_oracle
        )
        agent.enable_self_reflection()
    else:
        print("Using standard agent...")
        agent = Agent(
            cognitive_core=cognitive_core, 
            tool_registry=tool_registry, 
            db_path=db_path,
            rlhf_oracle=rlhf_oracle
        )

    # 5. Start the agent's main loop with an initial goal
    initial_observation = {
        "source_tool": "system_start",
        "data": {"text_data": "The AGI system has been activated. Your goal is to list the files in the 'src' directory."},
        "is_error": False
    }

    try:
        agent.run_main_loop(initial_observation)
        
        # If using consciousness agent, display final self-model
        if use_consciousness and hasattr(agent, 'get_self_model'):
            print("\nFinal self-model summary:")
            self_model = agent.get_self_model()
            print(f"Self-awareness metrics: {self_model.get('self_awareness_metrics', {})}")
            print(f"Number of cognitive states recorded: {len(agent.consciousness_core.self_model.cognitive_states)}")
    except KeyboardInterrupt:
        print("\nAgent loop terminated by user.")
        
        # If using consciousness agent, display final self-model
        if use_consciousness and hasattr(agent, 'get_self_model'):
            print("\nFinal self-model summary at termination:")
            self_model = agent.get_self_model()
            print(f"Self-awareness metrics: {self_model.get('self_awareness_metrics', {})}")
            print(f"Number of cognitive states recorded: {len(agent.consciousness_core.self_model.cognitive_states)}")

if __name__ == "__main__":
    main()