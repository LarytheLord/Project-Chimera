# Main entry point for the AGI system.

import sys
import os
import json

# Add the project root to the Python path to resolve imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.cognitive_core.interfaces import CognitiveCore
from src.agent.tool_user import ToolRegistry, WebSearchTool
from src.agent.agent import Agent

# --- Mock Implementation for Testing ---

class MockCognitiveCore(CognitiveCore):
    """A mock implementation of the Cognitive Core for testing the agent loop."""
    def load_model(self, model_path: str):
        print(f"Mock model loaded from {model_path}")

    def generate_response(self, inputs: dict[str, any]) -> str:
        # This is where the magic happens in a real model.
        # For now, we'll return a pre-defined action to test the loop.
        print("\n--- Cognitive Core received prompt: ---")
        print(inputs.get("text_data"))
        print("--- End of prompt ---\n")
        action = {
            "tool_name": "web_search",
            "arguments": {"query": "latest AGI research"}
        }
        return json.dumps(action)

    def train(self, dataset: any):

        print("Mock model is being trained.")

    def get_state(self) -> any:
        return {"mock_weights": [1, 2, 3]}

# --- Main Application ---

def main():
    """Initializes and runs the AGI agent."""
    print("Initializing Project Chimera...")

    # 1. Set up the Cognitive Core
    cognitive_core = MockCognitiveCore()
    cognitive_core.load_model("/path/to/mock/weights")

    # 2. Set up the Tool Registry
    tool_registry = ToolRegistry()
    tool_registry.register_tool(WebSearchTool())

    # 3. Create the Agent
    agent = Agent(cognitive_core=cognitive_core, tool_registry=tool_registry)

    # 4. Start the agent's main loop with an initial goal
    initial_observation = {
        "source_tool": "system_start",
        "data": {"text_data": "The AGI system has been activated. Your goal is to research the latest advancements in AGI."},
        "is_error": False
    }

    try:
        agent.run_main_loop(initial_observation)
    except KeyboardInterrupt:
        print("\nAgent loop terminated by user.")

if __name__ == "__main__":
    main()
