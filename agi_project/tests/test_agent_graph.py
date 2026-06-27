# System import
import os
import sys
from uuid import uuid4
from typing import TypedDict, Annotated, Sequence

# LLM import
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Tools import
from langchain_core.messages import BaseMessage,SystemMessage,ToolMessage,AIMessage,HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode



#local import
from agi_project.src.chimera.agent.memory import WorkingMemory,VectorEpisodicMemory, Experience
from agi_project.src.chimera.agent.tool_user import FileSystemTool
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]


_fs = FileSystemTool()

@tool
def read_file(path:str) -> str:
    """This function read the file
        Args: File path"""
    return _fs(operation="read_file",path=path)

@tool
def list_directory(path:str) -> str:
    """This function list_directory
        Args: parent directory path"""
    return _fs(operation="list_directory",path=path)
tools = [read_file,list_directory]

llm = ChatOpenRouter(
    model="poolside/laguna-xs.2:free", temperature = 0,max_tokens=500).bind_tools(tools)

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    systempompt = SystemMessage(
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = llm.invoke([systempompt]+state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"
#nodes in graph
tool_node = ToolNode(tools=tools)
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_node("tools", tool_node)

#connection of graph
graph.add_edge(START, "process")
graph.add_conditional_edges(
    "process",
    should_continue,{
        "continue":"tools",
        "end":END
    }

)
graph.add_edge("process","tools") 
agent = graph.compile()


m = WorkingMemory()
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
            #conver to appropriate BaseMassage
            if message[0]=="user":
                m.add(HumanMessage(content=message[1]))
            else:
                m.add(AIMessage(content=message[1]))
        else:
            message.pretty_print()
            m.add(message)


user_input_str = ""
while user_input_str.lower() != "exit":
    user_input_str = input("Enter: ")
    if user_input_str.lower() == "exit":
        break
    # Add the current input as HummanMessage to working memory
    new_user_message = HumanMessage(content=user_input_str)
    m.add(new_user_message)

    # Pass Message  history 
    input_for_agent = {"messages": m.get_context()}
    print_stream(agent.stream(input_for_agent, stream_mode="values"))

# Ask whether to save the session into VectorEpisodicMemory
save_choice = input("Save session to vector DB? (y/n): ").strip().lower()
if save_choice == "y":
    persist_dir = os.path.join(ROOT, "chroma_persist")
    vm = VectorEpisodicMemory(persist_path=persist_dir, collection_name="sessions")

    for idx, message in enumerate(m.get_context()):
        role = "human" if message[-1]=="User" else "ai"
        obs = {"role": role, "content": message.content}
        action = {"type": "message", "index": idx}
        outcome = {}
        exp = Experience(observation=obs, action=action, outcome=outcome)
        try:
            vm.remember(exp, doc_id=str(uuid4()))
        except Exception as e:
            print("Failed to remember experience:", e)

    print(f"Session saved to Chroma at: {vm.persist_path}")
