import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from cognitive_core.prometheus_core import PrometheusCognitiveCore
from agent.tool_user import ToolRegistry, WebSearchTool
from agent.agent import Agent
from rlhf.oracle import RLHFOracle

# --- Main Application ---

def main():
    """Initializes and runs the AGI agent."""
    print("Initializing Project Chimera with the Prometheus Engine and RLHF Oracle...")

    # 1. Set up the Cognitive Core
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

    # 3. Set up the RLHF Oracle
    try:
        reward_model_path = os.path.join(project_root, "trained_models", "reward_model")
        rlhf_oracle = RLHFOracle(model_path=reward_model_path)
    except FileNotFoundError as e:
        print(f"\n--- CONFIGURATION ERROR ---")
        print(f"Error initializing the RLHF Oracle: {e}")
        print("Please make sure you have trained a reward model by running scripts/train_reward_model.py first.")
        print("---")
        return

    # 4. Create the Agent
    db_path = os.path.join(project_root, "memory_db")
    agent = Agent(
        cognitive_core=cognitive_core, 
        tool_registry=tool_registry, 
        db_path=db_path,
        rlhf_oracle=rlhf_oracle
    )

    # 5. Start the agent's main loop with an initial goal
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