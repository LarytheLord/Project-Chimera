# This file will define the interface for using tools.

import abc
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass(frozen=True)
class ToolPolicy:
    """Execution policy metadata for a tool."""

    risk_level: str = "medium"
    capabilities: tuple[str, ...] = ()
    allowed_by_default: bool = True
    requires_approval: bool = False
    policy_note: str = ""

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Returns a JSON-serializable view of the policy."""
        return {
            "risk_level": self.risk_level,
            "capabilities": list(self.capabilities),
            "allowed_by_default": self.allowed_by_default,
            "requires_approval": self.requires_approval,
            "policy_note": self.policy_note,
        }

class Tool(abc.ABC):
    """Abstract Base Class for a tool that the agent can use."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The unique name of the tool."""
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """A description of what the tool does, for the agent to understand its purpose."""
        pass

    @abc.abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Returns a JSON schema describing the tool's arguments."""
        pass

    @abc.abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """Executes the tool with the given arguments."""
        pass

    def get_policy(self) -> ToolPolicy:
        """Returns execution policy metadata for the tool."""
        return ToolPolicy()

    def get_prompt_schema(self) -> Dict[str, Any]:
        """Returns the schema enriched with policy metadata for prompt rendering."""
        schema = dict(self.get_schema())
        schema["x-tool-policy"] = self.get_policy().to_prompt_dict()
        return schema

class ToolRegistry:
    """A registry that holds and provides access to all available tools."""

    def __init__(self, allow_restricted_tools: bool = False):
        self._tools: Dict[str, Tool] = {}
        self.allow_restricted_tools = allow_restricted_tools

    def register_tool(self, tool: Tool):
        """Registers a new tool."""
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' already registered.")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool:
        """Retrieves a tool by its name."""
        if name not in self._tools:
            raise ValueError(f"Tool with name '{name}' not found.")
        return self._tools[name]

    def unregister_tool(self, name: str):
        """Unregisters a tool."""
        if name not in self._tools:
            raise ValueError(f"Tool with name '{name}' not found.")
        del self._tools[name]

    def get_tool_schemas(self, include_restricted: bool | None = None) -> str:
        """Returns a JSON string of all prompt-visible tool schemas."""
        if include_restricted is None:
            include_restricted = self.allow_restricted_tools

        schemas = {
            name: tool.get_prompt_schema()
            for name, tool in self._tools.items()
            if include_restricted or tool.get_policy().allowed_by_default
        }
        return json.dumps(schemas, indent=2)

    def get_tool_names(self) -> List[str]:
        """Returns a list of all registered tool names."""
        return list(self._tools.keys())

    def execute_tool(
        self,
        name: str,
        arguments: Dict[str, Any] | None = None,
        allow_restricted: bool | None = None,
    ) -> Dict[str, Any]:
        """Executes a tool with policy checks and structured outcomes."""
        arguments = arguments or {}
        if not isinstance(arguments, dict):
            return self._build_outcome(
                source_tool=name,
                text_data="Tool arguments must be a JSON object.",
                is_error=True,
                status="invalid_arguments",
            )

        try:
            tool = self.get_tool(name)
        except ValueError as exc:
            return self._build_outcome(
                source_tool=name,
                text_data=str(exc),
                is_error=True,
                status="unknown_tool",
            )

        policy = tool.get_policy()
        if allow_restricted is None:
            allow_restricted = self.allow_restricted_tools

        if not policy.allowed_by_default and not allow_restricted:
            note = policy.policy_note or "This tool requires an explicit policy override."
            return self._build_outcome(
                source_tool=tool.name,
                text_data=f"Tool '{tool.name}' is restricted by policy. {note}",
                is_error=True,
                status="blocked",
                tool=tool,
            )

        try:
            result = tool(**arguments)
            return self._build_outcome(
                source_tool=tool.name,
                text_data=result,
                is_error=False,
                status="executed",
                tool=tool,
            )
        except Exception as exc:
            return self._build_outcome(
                source_tool=tool.name,
                text_data=str(exc),
                is_error=True,
                status="execution_error",
                tool=tool,
            )

    def execute_action(self, action: Any, allow_restricted: bool | None = None) -> Dict[str, Any]:
        """Executes a structured agent action via the policy-aware tool path."""
        if not isinstance(action, dict):
            return self._build_outcome(
                source_tool="unknown_tool",
                text_data="Agent action must be a JSON object.",
                is_error=True,
                status="invalid_action",
            )

        tool_name = action.get("tool_name")
        arguments = action.get("arguments", {})
        if not isinstance(tool_name, str) or not tool_name.strip():
            return self._build_outcome(
                source_tool="unknown_tool",
                text_data="Agent action is missing a valid 'tool_name'.",
                is_error=True,
                status="invalid_action",
            )

        return self.execute_tool(tool_name, arguments, allow_restricted=allow_restricted)

    def _build_outcome(
        self,
        source_tool: str,
        text_data: Any,
        is_error: bool,
        status: str,
        tool: Tool | None = None,
    ) -> Dict[str, Any]:
        policy = {"status": status}
        if tool is not None:
            policy.update(tool.get_policy().to_prompt_dict())
        return {
            "source_tool": source_tool,
            "data": {"text_data": text_data},
            "is_error": is_error,
            "policy": policy,
        }

# --- Tool Implementations ---

class WebSearchTool(Tool):
    """A tool for searching the web and scraping content."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Searches the web for a given query, scrapes the content of the top results, and returns the text."

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        }

    def get_policy(self) -> ToolPolicy:
        return ToolPolicy(
            risk_level="medium",
            capabilities=("web.search", "web.scrape"),
            allowed_by_default=True,
            requires_approval=False,
            policy_note="Use for open-web research and summarization tasks.",
        )

    def __call__(self, query: str) -> str:
        print(f"--- EXECUTING WEB SEARCH AND SCRAPE: {query} ---")
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                search_results = [r['href'] for r in ddgs.text(query, max_results=3)]
            scraped_content = []
            for url in search_results:
                scraped_content.append(self._scrape_page(url))
            return f"Results for '{query}':\n" + "\n".join(scraped_content)
        except Exception as e:
            return f"Error during web search and scrape: {e}"

    def _scrape_page(self, url: str) -> str:
        try:
            import requests
            from bs4 import BeautifulSoup

            response = requests.get(url, timeout=5)
            response.raise_for_status()  # Raise an exception for bad status codes
            soup = BeautifulSoup(response.content, 'html.parser')
            # Get text and remove extra whitespace
            text = ' '.join(soup.get_text().split())
            return f"Scraped content from {url}:\n{text[:1000]}..." # Return first 1000 chars
        except Exception as e:
            return f"Error scraping {url}: {e}"

class FileSystemTool(Tool):
    """A tool for interacting with the local file system."""

    @property
    def name(self) -> str:
        return "file_system"

    @property
    def description(self) -> str:
        return "Performs file system operations like listing directories and reading files."

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform.",
                        "enum": ["list_directory", "read_file"]
                    },
                    "path": {
                        "type": "string",
                        "description": "The path to the file or directory."
                    }
                },
                "required": ["operation", "path"]
            }
        }

    def get_policy(self) -> ToolPolicy:
        return ToolPolicy(
            risk_level="high",
            capabilities=("filesystem.list", "filesystem.read"),
            allowed_by_default=False,
            requires_approval=True,
            policy_note="Reads from the host file system and should stay gated unless explicitly enabled.",
        )

    def __call__(self, operation: str, path: str) -> str:
        try:
            if operation == "list_directory":
                return self._list_directory(path)
            elif operation == "read_file":
                return self._read_file(path)
            else:
                return f"Error: Unknown operation '{operation}'."
        except Exception as e:
            return f"Error performing file system operation: {e}"

    def _list_directory(self, path: str) -> str:
        if not os.path.isdir(path):
            return f"Error: Path '{path}' is not a valid directory."
        files = os.listdir(path)
        return json.dumps(files)

    def _read_file(self, path: str) -> str:
        if not os.path.isfile(path):
            return f"Error: Path '{path}' is not a valid file."
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
