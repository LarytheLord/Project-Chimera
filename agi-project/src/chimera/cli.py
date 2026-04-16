import argparse
import os
import sys

from chimera.agent.agent import Agent
from chimera.agent.tool_user import FileSystemTool, ToolRegistry, WebSearchTool
from chimera.cognitive_core.prometheus_core import PrometheusCognitiveCore
from chimera.consciousness.conscious_agent import ConsciousnessAwareAgent


def main():
    parser = argparse.ArgumentParser(description="Project Chimera CLI")
    parser.add_argument("--task", type=str, help="Initial task for the agent")
    parser.add_argument("--conscious", action="store_true", help="Enable consciousness (self-reflection)")
    parser.add_argument("--db-path", type=str, default="./chimera_db", help="Path to the memory database")
    
    args = parser.parse_args()
    
    if not args.task:
        print("Error: --task is required.")
        parser.print_help()
        sys.exit(1)
        
    # Check for API key
    if "CHIMERA_LLM_API_KEY" not in os.environ:
        print("Warning: CHIMERA_LLM_API_KEY not found in environment. Prometheus core may fail.")

    core = PrometheusCognitiveCore()
    tools = ToolRegistry()
    tools.register_tool(WebSearchTool())
    tools.register_tool(FileSystemTool())
    
    if args.conscious:
        print("--- Initializing Consciousness-Aware Agent ---")
        agent = ConsciousnessAwareAgent(
            cognitive_core=core,
            tool_registry=tools,
            db_path=args.db_path,
        )
        agent.enable_self_reflection()
    else:
        print("--- Initializing Standard Agent ---")
        agent = Agent(
            cognitive_core=core,
            tool_registry=tools,
            db_path=args.db_path,
        )
        
    print(f"--- Starting Agent Loop with Task: {args.task} ---")
    try:
        agent.run_main_loop({"task": args.task})
    except KeyboardInterrupt:
        print("\n--- Agent loop stopped by user ---")

if __name__ == "__main__":
    main()
