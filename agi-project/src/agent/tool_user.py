# This file will define the interface for using tools.

import abc
import os
import json
from typing import Dict, Any, List
from googlesearch import search

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

    def get_tool_schemas(self) -> str:
        """Returns a JSON string of all tool schemas."""
        schemas = {name: tool.get_schema() for name, tool in self._tools.items()}
        return json.dumps(schemas, indent=2)

# --- Tool Implementations ---

class WebSearchTool(Tool):
    """A tool for searching the web."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Searches the web for a given query and returns the top results."

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

    def __call__(self, query: str) -> str:
        print(f"--- EXECUTING WEB SEARCH: {query} ---")
        try:
            search_results = search(query, num=5, stop=5, pause=2)
            results = [str(r) for r in search_results]
            return f"Results for '{query}':\n" + "\n".join(results)
        except Exception as e:
            return f"Error during web search: {e}"

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
