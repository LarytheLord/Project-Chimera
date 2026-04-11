# Standalone CLI entry point for Project Chimera
# Usage: python -m chimera.main

import argparse
import json
import sys
import os
from typing import Optional

# Add the src directory to the path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chimera.agent.agent import Agent
from chimera.agent.tool_user import ToolRegistry, WebSearchTool, FileSystemTool
from chimera.cognitive_core.prometheus_core import PrometheusCognitiveCore
from chimera.consciousness.conscious_agent import ConsciousnessAwareAgent
from chimera.consciousness.narcissus_core import NarcissusConsciousnessCore


def create_agent(
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    enable_consciousness: bool = False,
    db_path: str = "./chimera_memory.lance"
) -> Agent:
    """Create a fully configured agent."""
    # Setup cognitive core
    core = PrometheusCognitiveCore(
        api_url=api_url or os.getenv("CHIMERA_LLM_API_URL"),
        api_key=api_key or os.getenv("CHIMERA_LLM_API_KEY")
    )
    
    # Setup tool registry
    registry = ToolRegistry()
    registry.register_tool(WebSearchTool())
    registry.register_tool(FileSystemTool())
    
    # Create agent
    agent = Agent(
        cognitive_core=core,
        tool_registry=registry,
        db_path=db_path
    )
    
    return agent


def run_interactive(agent: Agent, initial_query: Optional[str] = None):
    """Run the agent in interactive mode."""
    print("=" * 60)
    print("Project Chimera - AGI Agent")
    print("=" * 60)
    print("Type 'exit' or 'quit' to stop")
    print("Type 'status' to see agent state")
    print("Type 'audit' to see execution audit")
    print("Type 'policies' to see tool policies")
    print("=" * 60)
    
    while True:
        try:
            query = initial_query or input("\n> ")
            initial_query = None  # Clear after first use
            
            if query.lower() in ["exit", "quit"]:
                print("Shutting down...")
                break
            
            if query.lower() == "status":
                print("\n--- Agent Status ---")
                print(f"Tools: {', '.join(agent.tool_registry.get_tool_names())}")
                print(f"Working Memory Size: {len(agent.working_memory.memory)}")
                print(f"Feature Flags: {json.dumps(agent.get_feature_flags(), indent=2)}")
                continue
            
            if query.lower() == "audit":
                print("\n--- Audit Summary ---")
                audit = agent.get_audit_summary()
                print(f"Total Executions: {audit['total_executions']}")
                print(f"Task Runs: {audit['total_task_runs']}")
                print(f"Completed: {audit['completed']}")
                print(f"Failed: {audit['failed']}")
                continue
            
            if query.lower() == "policies":
                print("\n--- Tool Policies ---")
                policies = agent.get_tool_policies()
                print(json.dumps(policies, indent=2))
                continue
            
            if query.lower() == "opt-in":
                # Show available tools for opt-in
                print("\n--- Available Tools for Opt-In ---")
                for tool_name in agent.tool_registry.get_tool_names():
                    policy = agent.policy_registry.get_policy(tool_name)
                    if policy and policy.requires_opt_in:
                        opted_in = agent.policy_registry.is_tool_opted_in(tool_name)
                        print(f"  {tool_name}: {'✓' if opted_in else '✗'}")
                tool = input("Tool to opt-in: ")
                agent.opt_in_tool(tool)
                continue
            
            if query.lower() == "block":
                # Show tools that can be blocked
                print("\n--- Blockable Tools ---")
                for tool_name in agent.tool_registry.get_tool_names():
                    print(f"  {tool_name}")
                tool = input("Tool to block: ")
                agent.block_tool(tool)
                continue
            
            # Run the agent
            print("\n--- Running Agent ---")
            agent.run_main_loop(query)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Shutting down...")
            break
        except EOFError:
            print("\n\nEOF detected. Shutting down...")
            break


def run_single_query(agent: Agent, query: str):
    """Run the agent for a single query (non-interactive mode)."""
    print(f"Processing: {query}")
    agent.run_main_loop(query)
    print("\n--- Execution Complete ---")
    audit = agent.get_audit_summary()
    print(f"Total Executions: {audit['total_executions']}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Project Chimera - AGI Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m chimera.main --interactive
  python -m chimera.main --query "What is machine learning?"
  python -m chimera.main --api-url "https://api.example.com" --api-key "your-key"
  python -m chimera.main --enable-consciousness
        """
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query and exit"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        help="LLM API URL (overrides CHIMERA_LLM_API_URL env var)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="LLM API key (overrides CHIMERA_LLM_API_KEY env var)"
    )
    
    parser.add_argument(
        "--enable-consciousness",
        action="store_true",
        help="Enable consciousness core integration"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="./chimera_memory.lance",
        help="Path to LanceDB database (default: ./chimera_memory.lance)"
    )
    
    args = parser.parse_args()
    
    # Create agent
    agent = create_agent(
        api_url=args.api_url,
        api_key=args.api_key,
        enable_consciousness=args.enable_consciousness,
        db_path=args.db_path
    )
    
    # Enable consciousness if requested
    if args.enable_consciousness:
        agent.feature_flags.enable("consciousness_integration")
        print("Consciousness integration enabled")
    
    # Run in appropriate mode
    if args.query:
        run_single_query(agent, args.query)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
