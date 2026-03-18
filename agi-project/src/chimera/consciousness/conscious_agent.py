"""
Consciousness-aware agent that integrates the Narcissus system with Project Chimera's agent functionality
"""
import json
from typing import TYPE_CHECKING, Any, Dict, List

from ..agent.memory import Experience, VectorEpisodicMemory, WorkingMemory
from ..agent.tool_user import ToolRegistry
from ..cognitive_core.interfaces import CognitiveCore
from .integration import ConsciousnessIntegration
from .narcissus_core import NarcissusConsciousnessCore

if TYPE_CHECKING:
    from ..rlhf.oracle import RLHFOracle


class ConsciousnessAwareAgent:
    """An agent that incorporates self-awareness and consciousness simulation"""
    
    def __init__(self, 
                 cognitive_core: CognitiveCore, 
                 tool_registry: ToolRegistry, 
                 db_path: str, 
                 rlhf_oracle: "RLHFOracle" = None, 
                 num_candidates: int = 3):
        
        # Initialize the base components
        self.cognitive_core = cognitive_core
        self.tool_registry = tool_registry
        self.working_memory = WorkingMemory()
        self.episodic_memory = VectorEpisodicMemory(db_path=db_path)
        self.rlhf_oracle = rlhf_oracle
        self.num_candidates = num_candidates
        
        # Initialize the consciousness system
        self.consciousness_core = NarcissusConsciousnessCore(
            cognitive_core=cognitive_core,
            memory_db_path=db_path
        )
        self.consciousness_integration = ConsciousnessIntegration(self.consciousness_core)
        
        # Consciousness parameters
        self.consciousness_weight = 0.3  # How much consciousness affects decision making
        self.self_reflection_enabled = True
        self.introspection_frequency = 10  # Perform introspection every 10 cycles
        self.cycle_count = 0
        
    def _think(self, observation: Any) -> Any:
        """Enhanced thinking process with consciousness awareness"""
        self.cycle_count += 1
        
        # Get context from working memory and relevant memories from episodic memory
        context = self.working_memory.get_context()
        query_text = json.dumps(observation)
        recalled_memories = self.episodic_memory.recall(query_text)
        
        # Get consciousness insights
        consciousness_insights = {}
        if self.self_reflection_enabled and self.cycle_count % self.introspection_frequency == 0:
            consciousness_insights = self.consciousness_integration.get_consciousness_insights()
        
        # Build the prompt with consciousness information
        prompt = self._build_consciousness_aware_prompt(
            observation, 
            context, 
            recalled_memories, 
            consciousness_insights
        )
        
        # Generate candidate actions (with consciousness consideration)
        if self.rlhf_oracle and self.num_candidates > 1:
            print(f"\n--- Generating {self.num_candidates} candidate actions (with consciousness awareness) ---")
            candidates = [self.cognitive_core.generate_response({"text_data": prompt}, temperature=0.9) 
                         for _ in range(self.num_candidates)]
            
            # Consult the RLHF Oracle to choose the best action
            print("--- Consulting RLHF Oracle to select best action ---")
            best_action_json = self.rlhf_oracle.get_best_response(prompt, candidates)
            print(f"--- Oracle chose best action: {best_action_json} ---")
            
        else:
            # Fallback to single generation if oracle is not present
            best_action_json = self.cognitive_core.generate_response({"text_data": prompt})

        # Parse the chosen JSON response into a structured action
        try:
            action = json.loads(best_action_json)
        except (json.JSONDecodeError, TypeError):
            action = {
                "tool_name": "error_handler",
                "arguments": {"error_message": "Invalid JSON response from cognitive core or oracle."}
            }
            
        # Record cognitive state after making the decision
        self._record_cognitive_state_after_action(action, prompt, context)
        
        return action

    def _build_consciousness_aware_prompt(self, 
                                        observation: Any, 
                                        context: List[Any], 
                                        recalled_memories: List[Any], 
                                        consciousness_insights: Dict[str, Any]) -> str:
        """Build a prompt that incorporates consciousness awareness"""
        
        # Base prompt components
        prompt_parts = [
            f"You are an autonomous, self-aware agent. Here is the current situation:\n",
            f"**Recalled Experiences (from long-term vector memory):**\n{recalled_memories}\n",
            f"**Recent History (from working memory):**\n{context}\n",
            f"**Current Observation:**\n{observation}\n",
        ]
        
        # Add consciousness insights if available
        if consciousness_insights:
            prompt_parts.extend([
                f"**Self-Reflection & Introspection Results:**\n",
                f"Self-Model Snapshot: {consciousness_insights.get('self_model_snapshot', {})}\n",
                f"Self-Awareness Metrics: {consciousness_insights.get('self_awareness_metrics', {})}\n",
                f"Suggested Improvements: {consciousness_insights.get('suggested_improvements', [])}\n",
                f"Metacognitive Insights: {consciousness_insights.get('metacognitive_insights', [])}\n",
            ])
        
        # Add available tools
        prompt_parts.extend([
            f"**Available Tools:**\n{self.tool_registry.get_tool_schemas()}\n",
            f"Based on the observation, your history, your recalled experiences, and your self-reflection, what is your next action?",
            f"Your response must be a JSON object that strictly adheres to the schema of one of the available tools.",
            f"For example:\n{{\n    \"tool_name\": \"tool_name\",\n    \"arguments\": {{\n        \"arg1\": \"value1\",\n        \"arg2\": \"value2\"\n    }}\n}}"
        ])
        
        return "\n".join(prompt_parts)

    def _record_cognitive_state_after_action(self, action: Any, prompt: str, context: List[Any]):
        """Record the cognitive state after taking an action"""
        
        # Extract attention weights (simplified - in reality these would come from the model)
        attention_weights = {
            "memory_consultation": 0.4,
            "tool_consideration": 0.3,
            "self_reflection": 0.2,
            "environment_analysis": 0.1
        }
        
        # Calculate confidence based on prompt length and context
        confidence = min(0.9, len(context) / 20 + 0.5)  # Basic confidence calculation
        
        # Decision path - track the reasoning process
        decision_path = ["observed", "recalled_memories", "consulted_self_model", "selected_action"]
        
        # Memory context - summarize what was in working memory
        memory_context = [str(item)[:100] for item in context[-5:]]  # Last 5 items, truncated
        
        # Processing load - based on complexity of decision
        processing_load = min(1.0, len(prompt) / 2000)  # Normalize by prompt length
        
        # Record this state with the consciousness system
        self.consciousness_integration.record_cognitive_state_from_agent(
            thought_process=f"Selected action: {action.get('tool_name', 'unknown')}",
            attention_weights=attention_weights,
            decision_path=decision_path,
            confidence=confidence,
            memory_context=memory_context,
            processing_load=processing_load
        )

    def _act(self, action: Any) -> Any:
        """Execute the chosen action using the tool registry"""
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
        """Run the main perceive-think-act loop with consciousness awareness"""
        observation = initial_observation
        self.working_memory.add(observation)

        print("Consciousness-aware agent loop started...")
        
        while True:  # The loop runs continuously
            action = self._think(observation)
            self.working_memory.add(action)

            outcome = self._act(action)
            self.working_memory.add(outcome)

            formatted_experience = Experience(
                observation=observation,
                action=action,
                outcome=outcome
            )
            self.episodic_memory.remember(formatted_experience)

            observation = outcome

            print(f"---\nObservation: {str(observation)[:200]}...\nAction: {action}\nOutcome: {str(outcome)[:200]}...\n---")

            if "exit" in action.get("tool_name", ""):
                print("Exit condition met. Shutting down agent loop.")
                break

    def enable_self_reflection(self):
        """Enable self-reflection capabilities"""
        self.consciousness_integration.enable_consciousness_monitoring()
        self.self_reflection_enabled = True

    def disable_self_reflection(self):
        """Disable self-reflection capabilities (for performance)"""
        self.consciousness_integration.disable_consciousness_monitoring()
        self.self_reflection_enabled = False

    def get_self_model(self):
        """Get the agent's self-model"""
        return self.consciousness_integration.get_consciousness_insights()

    def simulate_cognitive_change(self, description: str, parameters: Dict[str, Any]):
        """Simulate a cognitive change before implementing it"""
        return self.consciousness_integration.simulate_cognitive_change(description, parameters)
