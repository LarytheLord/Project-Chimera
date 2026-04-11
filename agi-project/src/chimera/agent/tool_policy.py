# Tool policy system for policy-aware execution
# Implements sensitivity levels, permission boundaries, and execution controls

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import json


class ToolSensitivity(str, Enum):
    """Sensitivity levels for tools."""
    SAFE = "safe"  # No user confirmation needed
    MODERATE = "moderate"  # Requires user confirmation
    SENSITIVE = "sensitive"  # Requires explicit opt-in and confirmation
    BLOCKED = "blocked"  # Cannot be executed


class ExecutionPolicy(str, Enum):
    """Execution policy for tool behavior."""
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DENY = "deny"
    LOG_ONLY = "log_only"  # Execute but log for review


@dataclass
class ToolPolicy:
    """Policy configuration for a tool."""
    tool_name: str
    sensitivity: ToolSensitivity = ToolSensitivity.MODERATE
    policy: ExecutionPolicy = ExecutionPolicy.REQUIRE_CONFIRMATION
    description: str = ""
    allowed_operations: Optional[List[str]] = None  # e.g., ["read_file", "list_directory"]
    blocked_operations: Optional[List[str]] = None  # e.g., ["delete_file", "execute_code"]
    requires_opt_in: bool = False
    max_executions: Optional[int] = None  # Rate limiting
    execution_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_allowed(self, operation: Optional[str] = None) -> bool:
        """Check if the tool is allowed to execute based on policy."""
        if self.sensitivity == ToolSensitivity.BLOCKED:
            return False
        if self.policy == ExecutionPolicy.DENY:
            return False
        if self.max_executions and self.execution_count >= self.max_executions:
            return False
        if operation and self.blocked_operations and operation in self.blocked_operations:
            return False
        if operation and self.allowed_operations and operation not in self.allowed_operations:
            return False
        return True
    
    def record_execution(self):
        """Record that this tool was executed."""
        self.execution_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "sensitivity": self.sensitivity.value,
            "policy": self.policy.value,
            "description": self.description,
            "allowed_operations": self.allowed_operations,
            "blocked_operations": self.blocked_operations,
            "requires_opt_in": self.requires_opt_in,
            "max_executions": self.max_executions,
            "execution_count": self.execution_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolPolicy":
        """Create a ToolPolicy from a dictionary."""
        return cls(
            tool_name=data["tool_name"],
            sensitivity=ToolSensitivity(data.get("sensitivity", "moderate")),
            policy=ExecutionPolicy(data.get("policy", "require_confirmation")),
            description=data.get("description", ""),
            allowed_operations=data.get("allowed_operations"),
            blocked_operations=data.get("blocked_operations"),
            requires_opt_in=data.get("requires_opt_in", False),
            max_executions=data.get("max_executions"),
            execution_count=data.get("execution_count", 0),
            metadata=data.get("metadata", {})
        )


class ToolPolicyRegistry:
    """Central registry for tool policies and execution controls."""
    
    def __init__(self):
        self._policies: Dict[str, ToolPolicy] = {}
        self._global_blocked: Set[str] = set()
        self._global_opt_in: Set[str] = set()
    
    def register_policy(self, policy: ToolPolicy):
        """Register a policy for a tool."""
        self._policies[policy.tool_name] = policy
    
    def get_policy(self, tool_name: str) -> Optional[ToolPolicy]:
        """Get the policy for a tool."""
        return self._policies.get(tool_name)
    
    def update_policy(self, tool_name: str, **kwargs):
        """Update an existing policy."""
        if tool_name in self._policies:
            policy = self._policies[tool_name]
            for key, value in kwargs.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
    
    def block_tool(self, tool_name: str):
        """Globally block a tool."""
        self._global_blocked.add(tool_name)
        if tool_name in self._policies:
            self._policies[tool_name].sensitivity = ToolSensitivity.BLOCKED
            self._policies[tool_name].policy = ExecutionPolicy.DENY
    
    def unblock_tool(self, tool_name: str):
        """Unblock a tool."""
        self._global_blocked.discard(tool_name)
        if tool_name in self._policies:
            self._policies[tool_name].policy = ExecutionPolicy.REQUIRE_CONFIRMATION
    
    def opt_in_tool(self, tool_name: str):
        """Opt-in to a sensitive tool."""
        self._global_opt_in.add(tool_name)
    
    def opt_out_tool(self, tool_name: str):
        """Opt-out of a sensitive tool."""
        self._global_opt_in.discard(tool_name)
    
    def is_tool_opted_in(self, tool_name: str) -> bool:
        """Check if a tool is opted in."""
        return tool_name in self._global_opt_in
    
    def check_execution_permission(
        self,
        tool_name: str,
        operation: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Check if a tool can be executed.
        Returns: (allowed, reason)
        """
        # Check global block list
        if tool_name in self._global_blocked:
            return False, f"Tool '{tool_name}' is globally blocked"
        
        # Get policy
        policy = self._policies.get(tool_name)
        if policy is None:
            return True, "No policy defined - defaulting to allow"
        
        # Check sensitivity
        if policy.sensitivity == ToolSensitivity.BLOCKED:
            return False, f"Tool '{tool_name}' is blocked due to sensitivity"
        
        # Check opt-in requirement
        if policy.requires_opt_in and not self.is_tool_opted_in(tool_name):
            return False, f"Tool '{tool_name}' requires explicit opt-in"
        
        # Check operation-level permissions
        if operation:
            if policy.blocked_operations and operation in policy.blocked_operations:
                return False, f"Operation '{operation}' is blocked for tool '{tool_name}'"
            if policy.allowed_operations and operation not in policy.allowed_operations:
                return False, f"Operation '{operation}' is not allowed for tool '{tool_name}'"
        
        # Check rate limiting
        if policy.max_executions and policy.execution_count >= policy.max_executions:
            return False, f"Tool '{tool_name}' has reached max execution limit"
        
        return True, "Execution allowed"
    
    def record_execution(self, tool_name: str):
        """Record a tool execution for auditing."""
        if tool_name in self._policies:
            self._policies[tool_name].record_execution()
    
    def get_all_policies(self) -> Dict[str, ToolPolicy]:
        """Get all policies."""
        return self._policies.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize all policies."""
        return {
            name: policy.to_dict()
            for name, policy in self._policies.items()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolPolicyRegistry":
        """Deserialize policies."""
        registry = cls()
        for name, policy_data in data.items():
            registry.register_policy(ToolPolicy.from_dict(policy_data))
        return registry
