
import json
import uuid
from typing import TYPE_CHECKING, Any

from ..cognitive_core.interfaces import CognitiveCore
from .memory import Experience, VectorEpisodicMemory, WorkingMemory
from .tool_user import ToolRegistry, WebSearchTool
from .tool_policy import ToolPolicyRegistry, ToolPolicy, ToolSensitivity, ExecutionPolicy
from .provenance import AuditLogger, ExecutionRecord, TaskRun
from .feature_flags import FeatureFlagManager

if TYPE_CHECKING:
    from ..rlhf.oracle import RLHFOracle

class Agent:
    """The main agent class that orchestrates the AGI's operation."""

    def __init__(self, cognitive_core: CognitiveCore, tool_registry: ToolRegistry, db_path: str, rlhf_oracle: "RLHFOracle" = None, num_candidates: int = 3):
        self.cognitive_core = cognitive_core
        self.tool_registry = tool_registry
        self.working_memory = WorkingMemory()
        self.episodic_memory = VectorEpisodicMemory(db_path=db_path)
        self.rlhf_oracle = rlhf_oracle
        self.num_candidates = num_candidates
        
        # Initialize policy-aware execution system
        self.policy_registry = ToolPolicyRegistry()
        self.audit_logger = AuditLogger()
        self.feature_flags = FeatureFlagManager()
        
        # Initialize default tool policies
        self._init_default_policies()
        
        # Ensure WebSearchTool is registered
        if "web_search" not in self.tool_registry.get_tool_names():
            self.tool_registry.register_tool(WebSearchTool())
    
    def _init_default_policies(self):
        """Initialize default policies for tools."""
        # Web search - moderate sensitivity
        self.policy_registry.register_policy(ToolPolicy(
            tool_name="web_search",
            sensitivity=ToolSensitivity.MODERATE,
            policy=ExecutionPolicy.ALLOW,
            description="Web search and scraping tool",
            blocked_operations=["execute_code"],
            max_executions=50  # Rate limiting
        ))
        
        # File system - sensitive, read-only by default
        self.policy_registry.register_policy(ToolPolicy(
            tool_name="file_system",
            sensitivity=ToolSensitivity.SENSITIVE,
            policy=ExecutionPolicy.REQUIRE_CONFIRMATION,
            description="File system operations tool",
            allowed_operations=["list_directory", "read_file"],
            blocked_operations=["write_file", "delete_file", "execute_code"],
            requires_opt_in=True,
            max_executions=100
        ))

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
        """Executes the chosen action using the tool registry with policy enforcement and provenance tracking."""
        tool_name = action.get("tool_name", "unknown_tool")
        arguments = action.get("arguments", {})
        run_id = str(uuid.uuid4())
        start_time = None
        outcome = None
        
        try:
            # Check policy if feature flag is enabled
            if self.feature_flags.is_enabled("policy_aware_execution"):
                allowed, reason = self.policy_registry.check_execution_permission(
                    tool_name,
                    operation=arguments.get("operation")
                )
                if not allowed:
                    print(f"--- POLICY DENIED: {reason} ---")
                    outcome = {
                        "source_tool": tool_name,
                        "data": {"text_data": f"Action denied by policy: {reason}"},
                        "is_error": True
                    }
                    # Log the denied execution
                    if self.feature_flags.is_enabled("provenance_tracking"):
                        record = ExecutionRecord(
                            run_id=run_id,
                            tool_name=tool_name,
                            inputs=arguments,
                            error=reason,
                            success=False
                        )
                        self.audit_logger.log_execution(record)
                    return outcome
            
            # Execute the tool
            tool = self.tool_registry.get_tool(tool_name)
            
            # Record execution start
            if self.feature_flags.is_enabled("provenance_tracking"):
                start_time = __import__('datetime').datetime.now()
                self.policy_registry.record_execution(tool_name)
            
            result = tool(**arguments)
            outcome = {
                "source_tool": tool.name,
                "data": {"text_data": result},
                "is_error": False
            }
            
            # Log successful execution
            if self.feature_flags.is_enabled("provenance_tracking"):
                end_time = __import__('datetime').datetime.now()
                record = ExecutionRecord(
                    run_id=run_id,
                    tool_name=tool_name,
                    inputs=arguments,
                    outputs=result,
                    start_time=start_time,
                    end_time=end_time,
                    success=True
                )
                self.audit_logger.log_execution(record)
                
        except Exception as e:
            outcome = {
                "source_tool": tool_name,
                "data": {"text_data": str(e)},
                "is_error": True
            }
            # Log failed execution
            if self.feature_flags.is_enabled("provenance_tracking"):
                end_time = __import__('datetime').datetime.now()
                record = ExecutionRecord(
                    run_id=run_id,
                    tool_name=tool_name,
                    inputs=arguments,
                    error=str(e),
                    start_time=start_time,
                    end_time=end_time,
                    success=False
                )
                self.audit_logger.log_execution(record)
        
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
    
    def get_audit_summary(self) -> dict:
        """Get a summary of all audit logs."""
        return self.audit_logger.get_run_summary()
    
    def get_tool_policies(self) -> dict:
        """Get all tool policies."""
        return self.policy_registry.to_dict()
    
    def get_feature_flags(self) -> dict:
        """Get all feature flags."""
        return self.feature_flags.to_dict()
    
    def opt_in_tool(self, tool_name: str):
        """Opt-in to a sensitive tool."""
        self.policy_registry.opt_in_tool(tool_name)
        print(f"--- Opted-in to tool: {tool_name} ---")
    
    def block_tool(self, tool_name: str):
        """Block a tool from execution."""
        self.policy_registry.block_tool(tool_name)
        print(f"--- Blocked tool: {tool_name} ---")
