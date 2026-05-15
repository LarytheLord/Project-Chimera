"""Role: thin wrapper around a Chimera Agent with role-specific prompt and tool grants."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from ..agent.agent import Agent
from ..agent.tool_user import FileSystemTool, ToolRegistry
from ..cognitive_core.interfaces import CognitiveCore
from .prompts import build_role_prompt
from .work_order import WorkOrder


# Tools we know how to construct in Phase 1, keyed by their tool name.
# Web search is auto-registered by Agent.__init__ and only needs to be allowed/disallowed.
_TOOL_FACTORY = {
    "file_system": FileSystemTool,
}


class Role:
    """Base class for an org role. Subclasses set name, allowed_tools, artifact_key, next_role."""

    name: str = "Role"
    allowed_tools: tuple[str, ...] = ()
    artifact_key: str = "output"
    next_role: Optional[str] = None

    def __init__(
        self,
        cognitive_core: CognitiveCore,
        db_root: str,
        rlhf_oracle: Any = None,
    ) -> None:
        self.cognitive_core = cognitive_core

        registry = ToolRegistry()
        for tool_name in self.allowed_tools:
            factory = _TOOL_FACTORY.get(tool_name)
            if factory is not None:
                registry.register_tool(factory())

        db_path = os.path.join(db_root, self.name.lower())
        os.makedirs(db_path, exist_ok=True)

        self.agent = Agent(
            cognitive_core=cognitive_core,
            tool_registry=registry,
            db_path=db_path,
            rlhf_oracle=rlhf_oracle,
            num_candidates=1,
        )

        # Agent.__init__ auto-registers WebSearchTool. Remove it for roles that don't grant it.
        if (
            "web_search" not in self.allowed_tools
            and "web_search" in self.agent.tool_registry.get_tool_names()
        ):
            self.agent.tool_registry.unregister_tool("web_search")

    def process(self, wo: WorkOrder) -> WorkOrder:
        """Drive one role's contribution to the WorkOrder. Mutates and returns wo."""
        wo.begin()
        prompt = self._build_prompt(wo)
        raw = self.cognitive_core.generate_response({"text_data": prompt})
        parsed = self._parse_response(raw)
        return self._apply(wo, parsed)

    def _build_prompt(self, wo: WorkOrder) -> str:
        return build_role_prompt(
            role_name=self.name,
            goal=wo.goal,
            history=wo.history,
            artifacts=wo.artifacts,
            tools=self.agent.tool_registry.get_tool_names(),
        )

    def _parse_response(self, raw: str) -> dict:
        if raw is None:
            return {"output": {"error": "no response"}, "next_role": None}
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return {"output": {"error": "invalid JSON", "raw": raw}, "next_role": None}
        if not isinstance(parsed, dict):
            return {"output": {"error": "non-object JSON", "raw": raw}, "next_role": None}
        parsed.setdefault("output", {})
        parsed.setdefault("next_role", None)
        return parsed

    def _apply(self, wo: WorkOrder, parsed: dict) -> WorkOrder:
        output = parsed.get("output", {})
        # Subclasses can override next_role via parsed; otherwise fall back to class default.
        next_role = parsed.get("next_role") or self.next_role
        wo.artifacts[self.artifact_key] = output
        wo.advance(by_role=self.name, output=output, next_role=next_role)
        return wo
