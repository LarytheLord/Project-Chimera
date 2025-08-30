# Main entry point for the AGI system.

import sys
import os

# Add the project root to the Python path to resolve imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.cognitive_core.prometheus_core import PrometheusCognitiveCore
from src.agent.tool_user import ToolRegistry, WebSearchTool
from src.agent.agent import Agent

# --- Main Application ---

def main():
    """Initializes and runs the AGI agent."""
    print("Initializing Project Chimera with the Prometheus Engine...")

    # 1. Set up the Cognitive Core
    # IMPORTANT: You need to set the GEMINI_API_KEY environment variable for this to work.
    # For example, in your shell:
    # export GEMINI_API_KEY='your_api_key_here'
    #
    # You will also need to replace the api_url with the correct one for your model.
    try:
        cognitive_core = PrometheusCognitiveCore(
            api_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        )
        cognitive_core.load_model("gemini-pro")
    except ValueError as e:
        print(f"\n--- CONFIGURATION ERROR ---")
        print(f"Error initializing the cognitive core: {e}")
        print("Please make sure the GEMINI_API_KEY environment variable is set correctly.")
        print("---")
        return

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