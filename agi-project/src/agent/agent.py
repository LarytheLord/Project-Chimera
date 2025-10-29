
import json
from typing import Any, List

from cognitive_core.interfaces import CognitiveCore
from agent.memory import WorkingMemory, VectorEpisodicMemory, Experience
from agent.tool_user import ToolRegistry, Tool, WebSearchTool
from rlhf.oracle import RLHFOracle

class Agent:
    """The main agent class that orchestrates the AGI's operation."""

    def __init__(self, cognitive_core: CognitiveCore, tool_registry: ToolRegistry, db_path: str, rlhf_oracle: RLHFOracle = None, num_candidates: int = 3):
        self.cognitive_core = cognitive_core
        self.tool_registry = tool_registry
        self.working_memory = WorkingMemory()
        self.episodic_memory = VectorEpisodicMemory(db_path=db_path)
        self.rlhf_oracle = rlhf_oracle
        self.num_candidates = num_candidates
        # Ensure WebSearchTool is registered
        if "web_search" not in self.tool_registry.get_tool_names():
            self.tool_registry.register_tool(WebSearchTool())

    def _think(self, observation: Any) -> Any:
        """Uses the cognitive core and RLHF oracle to decide on the best next action."""
        # 1. Get context from working memory and relevant memories from episodic memory.
        context = self.working_memory.get_context()
        query_text = json.dumps(observation)
        recalled_memories = self.episodic_memory.recall(query_text)
        
        # 2. Format the prompt for the cognitive core.
        prompt = f"""
        You are an autonomous agent. Here is the current situation:
        
        **Recalled Experiences (from long-term vector memory):**
        {recalled_memories}

        **Recent History (from working memory):**
        {context}
        
        **Current Observation:**
        {observation}
        
        **Available Tools:**
        {self.tool_registry.get_tool_schemas()}
        
        Based on the observation, your history, and your recalled experiences, what is your next action?
        Your response must be a JSON object that strictly adheres to the schema of one of the available tools.
        For example:
        {{
            "tool_name": "tool_name",
            "arguments": {{
                "arg1": "value1",
                "arg2": "value2"
            }}
        }}
        """
        
        # 3. Generate multiple candidate actions.
        if self.rlhf_oracle and self.num_candidates > 1:
            print(f"\n--- Generating {self.num_candidates} candidate actions ---")
            candidates = [self.cognitive_core.generate_response({"text_data": prompt}, temperature=0.9) for _ in range(self.num_candidates)]
            
            # 4. Consult the RLHF Oracle to choose the best action.
            print("--- Consulting RLHF Oracle to select best action ---")
            best_action_json = self.rlhf_oracle.get_best_response(prompt, candidates)
            print(f"--- Oracle chose best action: {best_action_json} ---")
            
            # (Future Work): Here, we can implement the automated preference pair generation.
            # The `best_action_json` is the "chosen" response, and a random other candidate
            # can be the "rejected" one. This pair can be saved to preference_data.jsonl
            # to continuously improve the reward model.

        else:
            # Fallback to single generation if oracle is not present.
            best_action_json = self.cognitive_core.generate_response({"text_data": prompt})

        # 5. Parse the chosen JSON response into a structured action.
        try:
            action = json.loads(best_action_json)
        except (json.JSONDecodeError, TypeError):
            action = {
                "tool_name": "error_handler",
                "arguments": {"error_message": "Invalid JSON response from cognitive core or oracle."}
            }
        return action

    def _act(self, action: Any) -> Any:
        """Executes the chosen action using the tool registry."""
        try:
            tool_name = action["tool_name"]
            arguments = action["arguments"]
            tool = self.tool_registry.get_tool(tool_name)
            result = tool(**arguments)
            outcome = {"source_tool": tool.name, "data": {"text_data": result}, "is_error": False}
        except Exception as e:
            outcome = {"source_tool": action.get("tool_name", "unknown_tool"), "data": {"text_data": str(e)}, "is_error": True}
        return outcome

    def run_main_loop(self, initial_observation: Any):
        """Runs the main perceive-think-act loop of the agent."""
        observation = initial_observation
        self.working_memory.add(observation)

        while True: # The loop runs continuously
            action = self._think(observation)
            self.working_memory.add(action)

            outcome = self._act(action)
            self.working_memory.add(outcome)

            experience = Experience(observation=observation, action=action, outcome=outcome)
            self.episodic_memory.remember(experience)

            observation = outcome

            print(f"---Observation: {observation}\nAction: {action}\nOutcome: {outcome}---")

            if "exit" in action.get("tool_name", ""):
                print("Exit condition met. Shutting down agent loop.")
                break
