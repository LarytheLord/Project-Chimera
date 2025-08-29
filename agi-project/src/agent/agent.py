# This file will contain the main Agent class, which orchestrates the perception-action loop.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from typing import Any

from cognitive_core.interfaces import CognitiveCore
from .memory import WorkingMemory, EpisodicMemory, Experience
from .tool_user import ToolRegistry, Tool

# Placeholder for protobuf messages
# from protos import core_pb2

class Agent:
    """The main agent class that orchestrates the AGI's operation."""

    def __init__(self, cognitive_core: CognitiveCore, tool_registry: ToolRegistry, memory_filepath: str = None):
        self.cognitive_core = cognitive_core
        self.tool_registry = tool_registry
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory(filepath=memory_filepath)

    def _think(self, observation: Any) -> Any:
        """Uses the cognitive core to decide on the next action."""
        # 1. Get context from working memory and relevant memories from episodic memory.
        context = self.working_memory.get_context()
        recalled_memories = self.episodic_memory.recall(observation.get('data', {}).get('text_data', '')) # Recall based on observation text
        
        # 2. Format this into a prompt for the cognitive core.
        # This is a critical step. The prompt needs to instruct the model to act as an agent.
        prompt = f"""
        You are an autonomous agent. Here is the current situation:
        
        **Recalled Experiences (from long-term memory):**
        {recalled_memories}

        **Recent History (from working memory):**
        {context}
        
        **Current Observation:**
        {observation}
        
        **Available Tools:**
        {self.tool_registry.get_tool_descriptions()}
        
        Based on the observation, your history, and your recalled experiences, what is your next action?
        Your response must be a JSON object of the following format:
        {{
            "tool_name": "tool_to_call",
            "arguments": {{
                "arg1": "value1",
                "arg2": "value2"
            }}
        }}
        """
        
        # 3. Call the cognitive core to get the desired action.
        # The response from the model is expected to be a JSON string.
        response_json = self.cognitive_core.generate_response({"text_data": prompt})
        
        # 4. Parse the JSON response into a structured action.
        try:
            action = json.loads(response_json)
        except json.JSONDecodeError:
            # Handle cases where the model's output is not valid JSON
            # For now, we'll create an error action. A more sophisticated approach
            # could involve asking the model to correct its output.
            action = {
                "tool_name": "error_handler",
                "arguments": {"error_message": "Invalid JSON response from cognitive core."}
            }
        return action

    def _act(self, action: Any) -> Any:
        """Executes the chosen action using the tool registry."""
        try:
            tool_name = action["tool_name"]
            arguments = action["arguments"]
            tool = self.tool_registry.get_tool(tool_name)
            result = tool(**arguments)
            # Create a structured observation object
            outcome = {"source_tool": tool.name, "data": {"text_data": result}, "is_error": False}
        except Exception as e:
            outcome = {"source_tool": action.get("tool_name", "unknown_tool"), "data": {"text_data": str(e)}, "is_error": True}
        return outcome

    def run_main_loop(self, initial_observation: Any):
        """Runs the main perceive-think-act loop of the agent."""
        observation = initial_observation
        self.working_memory.add(observation)

        while True: # The loop runs continuously
            # 1. Think: Decide on the next action.
            action = self._think(observation)
            self.working_memory.add(action)

            # 2. Act: Execute the action.
            outcome = self._act(action)
            self.working_memory.add(outcome)

            # 3. Remember: Store the full experience.
            experience = Experience(observation=observation, action=action, outcome=outcome)
            self.episodic_memory.remember(experience)

            # The outcome of the last action becomes the new observation for the next loop.
            observation = outcome

            print(f"""---
Observation: {observation}
Action: {action}
Outcome: {outcome}
---""")

            # In a real system, there would be a condition to break the loop.
            # For now, we can add a simple check.
            if "exit" in action.get("tool_name", ""):
                print("Exit condition met. Shutting down agent loop.")
                break
