
import json
import sys
import os
from typing import Any, Dict



from agent.agent import Agent
from agent.tool_user import Tool, ToolRegistry
from cognitive_core.interfaces import CognitiveCore

# --- Mock Components ---

class MockCognitiveCore(CognitiveCore):
    """A mock cognitive core that returns a predefined JSON action."""
    def __init__(self, action_json: Dict[str, Any]):
        self.action_json = action_json

    def generate_response(self, prompt: Dict[str, str]) -> str:
        # In a real scenario, this would involve an LLM call.
        # Here, we just return the predefined action as a JSON string.
        return json.dumps(self.action_json)

def sum_numbers(a: int, b: int) -> int:
    """A simple tool that adds two numbers."""
    return a + b

# --- Test Fixture ---

def setup_agent_for_test():
    """Sets up the agent with mock components for a predictable test."""
    # 1. Define the mock action the LLM will be simulated to return.
    mock_action = {
        "tool_name": "sum_numbers",
        "arguments": {"a": 5, "b": 10}
    }
    
    # 2. Create the mock cognitive core.
    mock_core = MockCognitiveCore(action_json=mock_action)

    # 3. Create and register a simple tool.
    sum_tool = Tool(name="sum_numbers", func=sum_numbers, description="Adds two integers a and b.")
    tool_registry = ToolRegistry()
    tool_registry.register_tool(sum_tool)

    # 4. Instantiate the agent with the mock components.
    agent = Agent(cognitive_core=mock_core, tool_registry=tool_registry)
    
    return agent

# --- The Test ---

def test_agent_think_act_loop():
    """
    Tests one full cycle of the agent's perceive-think-act loop.
    It verifies that the agent correctly parses the mock LLM response,
    calls the appropriate tool with the correct arguments, and processes the result.
    """
    # 1. Setup
    agent = setup_agent_for_test()
    initial_observation = {"source": "user", "data": {"text_data": "What is 5 + 10?"}}
    
    # We need to manually step through the loop to inspect the intermediate objects.
    
    # 2. Think
    # The agent observes the initial input and thinks about the next action.
    agent.working_memory.add(initial_observation)
    action = agent._think(initial_observation)
    
    # Assert the action was parsed correctly
    expected_action = {
        "tool_name": "sum_numbers",
        "arguments": {"a": 5, "b": 10}
    }
    assert action == expected_action
    
    # 3. Act
    # The agent executes the action.
    agent.working_memory.add(action)
    outcome = agent._act(action)

    # Assert the outcome is correct
    expected_outcome = {
        'source_tool': 'sum_numbers', 
        'data': {'text_data': 15}, 
        'is_error': False
    }
    assert outcome == expected_outcome
    
    print("Agent think-act cycle test passed successfully!")

if __name__ == "__main__":
    test_agent_think_act_loop()

