import os
import sys
from typing import TypedDict, List, Union
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool

# Ensure src is importable when tests run directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from chimera.agent.memory import WorkingMemory, VectorEpisodicMemory, Experience

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenRouter(
    model="poolside/laguna-xs.2:free", temperature = 0,max_tokens=500)




def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content)) 
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"])

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()


m=WorkingMemory(100)
user_input = input("Enter: ")
while user_input != "exit":
    m.add(HumanMessage(content=user_input))
    result = agent.invoke({"messages": m.get_context()})
    m.history = result["messages"]
    user_input = input("Enter: ")
m.add(HumanMessage(content=user_input))

# with open("logging.txt", "w") as file:
#     file.write("Your Conversation Log:\n")
    
#     for message in m.get_context():
#         if isinstance(message, HumanMessage):
#             file.write(f"You: {message.content}\n")
#         elif isinstance(message, AIMessage):
#             file.write(f"AI: {message.content}\n\n")
#     file.write("End of Conversation")

# print("Conversation saved to logging.txt")

# Ask whether to save the session into VectorEpisodicMemory
save_choice = input("Save session to vector DB? (y/n): ").strip().lower()
if save_choice == "y":
    persist_dir = os.path.join(ROOT, "chroma_persist")
    vm = VectorEpisodicMemory(persist_path=persist_dir, collection_name="sessions")

    for idx, message in enumerate(m.get_context()):
        role = "human" if isinstance(message, HumanMessage) else "ai"
        obs = {"role": role, "content": message.content}
        action = {"type": "message", "index": idx}
        outcome = {}
        exp = Experience(observation=obs, action=action, outcome=outcome)
        try:
            vm.remember(exp, doc_id=str(uuid4()))
        except Exception as e:
            print("Failed to remember experience:", e)

    print(f"Session saved to Chroma at: {vm.persist_path}")
