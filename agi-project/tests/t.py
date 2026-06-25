import importlib.util
import pathlib
from langchain_core.tools import tool
# Load the tool_user module directly from its source file to avoid package imports
tool_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "chimera" / "agent" / "tool_user.py"
spec = importlib.util.spec_from_file_location("src.chimera.agent.tool_user", str(tool_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

FileSystemTool = mod.FileSystemTool

# Instantiate the tool (no ctor args) and call it with operation and path

fs = FileSystemTool()
result = fs("read_file", "C:\\Users\\PRIT TANDEL\\Desktop\\project\\Project-Chimera\\agi-project\\src\\chimera\\agent\\agent.py")  # Replace with a valid file path on your system
print(result)