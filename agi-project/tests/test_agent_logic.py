
import sys
import os


# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import json
from typing import Any, Dict


from agent.agent import Agent
from agent.memory import Experience
from agent.tool_user import Tool, ToolRegistry

from cognitive_core.interfaces import CognitiveCore

# --- Mock Components ---

class MockCognitiveCore(CognitiveCore):
    """A mock cognitive core that returns a predefined JSON action."""
    def __init__(self, action_json: Dict[str, Any]):
        self.action_json = action_json

    def generate_response(self, prompt: Dict[str, str]) -> str:
        return json.dumps(self.action_json)

    def load_model(self, model_path: str):
        pass

    def train(self, dataset: Any):
        pass

    def get_state(self) -> Any:
        pass
    @property
    def name(self) -> str:
        return "sum_numbers"

    @property
    def description(self) -> str:
        return "Adds two integers a and b."

    def __call__(self, a: int, b: int) -> Any:
        return a + b

# --- Test Fixture ---

def setup_agent_for_test(memory_filepath: str = None):
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
        memory_filepath=memory_filepath
    )
    return agent

# --- The Test ---

def test_agent_think_act_loop_and_memory_persistence():
    """
    Tests one full cycle of the agent's perceive-think-act loop and memory persistence.
    """
    MEMORY_FILE = "test_agent_memory.json"
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)

    # 1. Setup the first agent and run a cycle
    agent1 = setup_agent_for_test(memory_filepath=MEMORY_FILE)
    initial_observation = {"source": "user", "data": {"text_data": "What is 5 + 10?"}}
    
    # We are not testing the full loop here, but the components
    # Think
    action = agent1._think(initial_observation)
    expected_action = {
        "tool_name": "sum_numbers",
        "arguments": {"a": 5, "b": 10}
    }
    assert action == expected_action
    
    # Act
    outcome = agent1._act(action)
    expected_outcome = {
        'source_tool': 'sum_numbers',
        'data': {'text_data': 15},
        'is_error': False
    }
    assert outcome == expected_outcome

    # Remember
    experience = Experience(observation=initial_observation, action=action, outcome=outcome)
    agent1.episodic_memory.remember(experience)


    # 2. Create a second agent and verify memory is loaded
    agent2 = setup_agent_for_test(memory_filepath=MEMORY_FILE)
    
    # Verify that the second agent has loaded the history from the first agent.
    assert len(agent2.episodic_memory.experiences) == 1
    loaded_experience = agent2.episodic_memory.experiences[0]
    
    # We need to convert namedtuple to tuple for comparison
    assert tuple(loaded_experience) == experience

    # --- Cleanup ---
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    
    print("Agent think-act-memory cycle test passed successfully!")

if __name__ == "__main__":
    test_agent_think_act_loop_and_memory_persistence()





