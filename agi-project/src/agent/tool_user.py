# This file will define the interface for using tools.

import abc
from typing import Dict, Any, List

# Placeholder for protobuf messages
# from ..protos import core_pb2

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
    def __call__(self, args: Dict[str, Any]) -> Any:
        """Executes the tool with the given arguments."""
        pass

class ToolRegistry:
    """A registry that holds and provides access to all available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

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

    def get_tool_descriptions(self) -> str:
        """Returns a formatted string of all tool names and descriptions."""
        return "\n".join([f"- {name}: {tool.description}" for name, tool in self._tools.items()])

# --- Example Tool Implementation ---

class WebSearchTool(Tool):
    """A tool for searching the web."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Searches the web for a given query and returns the top results."

    def __call__(self, args: Dict[str, Any]) -> str:
        query = args.get("query")
        if not query:
            return "Error: Missing required argument 'query'."
        
        print(f"--- EXECUTING WEB SEARCH: {query} ---")
        # In a real implementation, this would call a search API.
        return f"Results for '{query}': 1. The sky is blue. 2. Water is wet."
