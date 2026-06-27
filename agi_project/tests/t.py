from agi_project.src.chimera.agent.tool_user import FileSystemTool
from langchain_core.tools import tool

# Instantiate the tool and call it with the desired operation and path.
_fs = FileSystemTool()
@tool
def read_file(path: str):
    """This function is use for read files
        Args:It take file path as argument"""
    return _fs(operation="read_file",path=path)
result = read_file.invoke({"path":"C:\\Users\\PRIT TANDEL\\Desktop\\project\\Project-Chimera\\agi_project\\src\\chimera\\agent\\agent.py"})
print(result)