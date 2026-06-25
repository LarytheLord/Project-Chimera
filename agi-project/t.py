from src.chimera import FileSystemTool

# Instantiate the tool and call it with the desired operation and path.
fs = FileSystemTool()
result = fs(operation="read_file", path="C:\\Users\\PRIT TANDEL\\Desktop\\project\\Project-Chimera\\agi-project\\src\\chimera\\agent\\agent.py")
print(result)