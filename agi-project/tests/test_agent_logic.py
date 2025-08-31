import sys
import os
import shutil

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import json
from typing import Any, Dict
import pytest

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

class SumTool(Tool):
    """A simple tool for testing that adds two numbers."""
    @property
    def name(self) -> str:
        return "sum_numbers"

    @property
    def description(self) -> str:
        return "Adds two integers a and b."

    def __call__(self, a: int, b: int) -> Any:
        return a + b

# --- Test Fixture ---

@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary directory for the LanceDB database."""
    return str(tmp_path)

def setup_agent_for_test(db_path: str):
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
        db_path=db_path
    )
    return agent

# --- The Test ---

def test_agent_vector_memory_and_recall(temp_db_path):
    """
    Tests the agent's ability to use the VectorEpisodicMemory to remember
    an experience and recall it based on semantic similarity.
    """
    # 1. Setup agent and a unique experience
    agent = setup_agent_for_test(db_path=temp_db_path)
    
    unique_observation = {"source": "user", "data": {"text_data": "The sky is filled with blue elephants today."}}
    action_taken = {"tool_name": "observe", "arguments": {"phenomenon": "flying blue elephants"}}
    outcome_result = {"source_tool": "observe", "data": {"text_data": "Confirmed: The elephants are indeed blue and airborne."}}
    
    experience_to_remember = Experience(
        observation=unique_observation, 
        action=action_taken, 
        outcome=outcome_result
    )

    # 2. Agent remembers the experience
    agent.episodic_memory.remember(experience_to_remember)

    # 3. Agent recalls memory with a semantically similar query
    # This query does not use the same keywords but has a similar meaning.
    semantic_query = "What can you tell me about unusually colored pachyderms in the air?"
    recalled_experiences = agent.episodic_memory.recall(semantic_query, top_k=1)

    # 4. Verification
    # Verify that the agent recalled the correct experience.
    assert len(recalled_experiences) == 1, "Should have recalled one experience."
    
    recalled_exp = recalled_experiences[0]
    
    # Compare the observation text which is the most unique part
    assert recalled_exp.observation["data"]["text_data"] == unique_observation["data"]["text_data"], \
        "The recalled experience should match the one that was remembered."

    print("Agent vector memory (remember and recall) test passed successfully!")

if __name__ == "__main__":
    # This allows running the test directly for debugging, though `pytest` is preferred.
    # To run, you need a temporary directory. Pytest handles this automatically.
    temp_dir = "./temp_test_db"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    try:
        test_agent_vector_memory_and_recall(temp_dir)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)