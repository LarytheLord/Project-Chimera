import argparse
import os
import sys

from .agent.agent import Agent
from .agent.tool_user import FileSystemTool, ToolRegistry, WebSearchTool
from .cognitive_core.prometheus_core import PrometheusCognitiveCore
from .consciousness.conscious_agent import ConsciousnessAwareAgent
from .eventbus import EventBusConfig, build_event_bus


def main():
    parser = argparse.ArgumentParser(description="Project Chimera CLI")
    parser.add_argument("--task", type=str, help="Initial task for the agent")
    parser.add_argument("--conscious", action="store_true", help="Enable consciousness (self-reflection)")
    parser.add_argument("--db-path", type=str, default="./chimera_db", help="Path to the memory database")
    parser.add_argument(
        "--kafka-brokers",
        type=str,
        default=None,
        help=(
            "Comma-separated Kafka/Redpanda bootstrap servers (e.g. localhost:9092). "
            "If omitted the agent uses CHIMERA_KAFKA_BROKERS or runs without an event bus."
        ),
    )
    parser.add_argument("--agent-id", type=str, default=None, help="Agent identifier embedded in events")
    parser.add_argument("--session-id", type=str, default=None, help="Session identifier embedded in events")

    args = parser.parse_args()

    if not args.task:
        print("Error: --task is required.")
        parser.print_help()
        sys.exit(1)

    # Check for API key
    if "CHIMERA_LLM_API_KEY" not in os.environ:
        print("Warning: CHIMERA_LLM_API_KEY not found in environment. Prometheus core may fail.")

    # Build event bus from env or CLI overrides
    bus_config = EventBusConfig.from_env()
    if args.kafka_brokers:
        bus_config.bootstrap_servers = [b.strip() for b in args.kafka_brokers.split(",") if b.strip()]
    if args.agent_id:
        bus_config.agent_id = args.agent_id
    if args.session_id:
        bus_config.session_id = args.session_id
    event_bus = build_event_bus(bus_config)
    print(
        f"--- Event bus: {type(event_bus).__name__} "
        f"(brokers={bus_config.bootstrap_servers or 'none'}) ---"
    )

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
            event_bus=event_bus,
            agent_id=bus_config.agent_id,
            session_id=bus_config.session_id,
        )
        agent.enable_self_reflection()
    else:
        print("--- Initializing Standard Agent ---")
        agent = Agent(
            cognitive_core=core,
            tool_registry=tools,
            db_path=args.db_path,
            event_bus=event_bus,
            agent_id=bus_config.agent_id,
            session_id=bus_config.session_id,
        )

    print(f"--- Starting Agent Loop with Task: {args.task} ---")
    try:
        agent.run_main_loop({"task": args.task})
    except KeyboardInterrupt:
        print("\n--- Agent loop stopped by user ---")
    finally:
        try:
            event_bus.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
