# This file will define the interface for using tools.

import abc
import json
import os
from typing import Any, Dict, List

from langchain_core.tools import tool

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

    def get_tool_names(self) -> List[str]:
        """Returns a list of all registered tool names."""
        return list(self._tools.keys())

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

# Terminal executation tools
class Terminal_exute(Tool):
    """This tool will execute all command and returns it output"""
    @property
    def name(self) -> str:
        return "Terminal_exute"

    @property
    def description(self) -> str:
        return "Performs execution of the command in terminal"
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
                    },
                    "path": {
                        "type": "string",
                        "description": "command"
                    }
                },
                "required": "command"
            }
        }
    def __call__(self,command: str) -> str:
        self.command = command.split(" ")
        import os
        import subprocess


        # To execute a shell command like npm install:
        result = subprocess.run(self.command,capture_output=True,text=True)
        print(result.stdout)
        if result.returncode != 0 :
            return f"\nError {result.stderr}"
        return result
    


