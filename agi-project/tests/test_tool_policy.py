# Tests for policy-aware tool execution system

import pytest
from chimera.agent.tool_policy import (
    ToolPolicy,
    ToolPolicyRegistry,
    ToolSensitivity,
    ExecutionPolicy
)


class TestToolPolicy:
    def test_default_policy(self):
        policy = ToolPolicy(tool_name="test_tool")
        assert policy.sensitivity == ToolSensitivity.MODERATE
        assert policy.policy == ExecutionPolicy.REQUIRE_CONFIRMATION
        assert policy.is_allowed() == True
    
    def test_blocked_policy(self):
        policy = ToolPolicy(
            tool_name="blocked_tool",
            sensitivity=ToolSensitivity.BLOCKED
        )
        assert policy.is_allowed() == False
    
    def test_deny_policy(self):
        policy = ToolPolicy(
            tool_name="denied_tool",
            policy=ExecutionPolicy.DENY
        )
        assert policy.is_allowed() == False
    
    def test_rate_limit_policy(self):
        policy = ToolPolicy(
            tool_name="limited_tool",
            max_executions=2
        )
        assert policy.is_allowed() == True
        policy.record_execution()
        assert policy.is_allowed() == True
        policy.record_execution()
        assert policy.is_allowed() == False  # Reached limit
    
    def test_operation_allowlist(self):
        policy = ToolPolicy(
            tool_name="restricted_tool",
            allowed_operations=["read", "list"]
        )
        assert policy.is_allowed("read") == True
        assert policy.is_allowed("write") == False
    
    def test_operation_blocklist(self):
        policy = ToolPolicy(
            tool_name="partially_restricted",
            blocked_operations=["delete", "execute"]
        )
        assert policy.is_allowed("read") == True
        assert policy.is_allowed("delete") == False
        assert policy.is_allowed("execute") == False
    
    def test_serialization(self):
        policy = ToolPolicy(
            tool_name="serialize_me",
            sensitivity=ToolSensitivity.SENSITIVE,
            policy=ExecutionPolicy.ALLOW,
            description="Test policy",
            allowed_operations=["read"],
            blocked_operations=["write"],
            requires_opt_in=True,
            max_executions=10
        )
        data = policy.to_dict()
        restored = ToolPolicy.from_dict(data)
        assert restored.tool_name == policy.tool_name
        assert restored.sensitivity == policy.sensitivity
        assert restored.policy == policy.policy
        assert restored.allowed_operations == policy.allowed_operations


class TestToolPolicyRegistry:
    def test_register_policy(self):
        registry = ToolPolicyRegistry()
        policy = ToolPolicy(tool_name="test_tool")
        registry.register_policy(policy)
        assert registry.get_policy("test_tool") == policy
    
    def test_block_tool(self):
        registry = ToolPolicyRegistry()
        policy = ToolPolicy(tool_name="test_tool")
        registry.register_policy(policy)
        registry.block_tool("test_tool")
        allowed, reason = registry.check_execution_permission("test_tool")
        assert allowed == False
        assert "blocked" in reason.lower()
    
    def test_unblock_tool(self):
        registry = ToolPolicyRegistry()
        policy = ToolPolicy(tool_name="test_tool")
        registry.register_policy(policy)
        registry.block_tool("test_tool")
        registry.unblock_tool("test_tool")
        allowed, reason = registry.check_execution_permission("test_tool")
        assert allowed == True
    
    def test_opt_in_requirement(self):
        registry = ToolPolicyRegistry()
        policy = ToolPolicy(
            tool_name="sensitive_tool",
            requires_opt_in=True
        )
        registry.register_policy(policy)
        allowed, reason = registry.check_execution_permission("sensitive_tool")
        assert allowed == False
        assert "opt-in" in reason.lower()
        registry.opt_in_tool("sensitive_tool")
        allowed, reason = registry.check_execution_permission("sensitive_tool")
        assert allowed == True
    
    def test_operation_check(self):
        registry = ToolPolicyRegistry()
        policy = ToolPolicy(
            tool_name="file_tool",
            allowed_operations=["read", "list"],
            blocked_operations=["delete"]
        )
        registry.register_policy(policy)
        allowed, reason = registry.check_execution_permission("file_tool", "read")
        assert allowed == True
        allowed, reason = registry.check_execution_permission("file_tool", "delete")
        assert allowed == False
    
    def test_rate_limiting(self):
        registry = ToolPolicyRegistry()
        policy = ToolPolicy(
            tool_name="limited_tool",
            max_executions=1
        )
        registry.register_policy(policy)
        allowed, reason = registry.check_execution_permission("limited_tool")
        assert allowed == True
        registry.record_execution("limited_tool")
        allowed, reason = registry.check_execution_permission("limited_tool")
        assert allowed == False
    
    def test_serialization(self):
        registry = ToolPolicyRegistry()
        registry.register_policy(ToolPolicy(
            tool_name="tool1",
            sensitivity=ToolSensitivity.SAFE
        ))
        registry.register_policy(ToolPolicy(
            tool_name="tool2",
            sensitivity=ToolSensitivity.SENSITIVE
        ))
        data = registry.to_dict()
        restored = ToolPolicyRegistry.from_dict(data)
        assert len(restored.get_all_policies()) == 2
        assert restored.get_policy("tool1").sensitivity == ToolSensitivity.SAFE
