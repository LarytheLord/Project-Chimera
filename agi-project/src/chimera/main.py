"""Standalone CLI entrypoint for Project Chimera."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .agent.agent import Agent
from .agent.memory import Experience
from .agent.tool_user import FileSystemTool, ToolRegistry, WebSearchTool
from .consciousness.conscious_agent import ConsciousnessAwareAgent

DEFAULT_DB_PATH = Path(os.environ.get("CHIMERA_DB_PATH", ".chimera")).expanduser()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for standalone Chimera usage."""
    parser = argparse.ArgumentParser(
        prog="chimera",
        description=(
            "Run Project Chimera as a standalone research agent. "
            "By default it exposes web search only; local file-system access stays opt-in."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("standard", "consciousness"),
        default="standard",
        help="Select the base agent loop or the consciousness-aware variant.",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="Directory used for LanceDB-backed episodic memory.",
    )
    parser.add_argument(
        "--prompt",
        help="Run a single prompt and exit. Without this flag, Chimera starts an interactive shell.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print each turn as JSON instead of a human-readable summary.",
    )
    parser.add_argument(
        "--allow-file-system",
        action="store_true",
        help="Register the local file-system tool for read-only exploration.",
    )
    return parser


def build_initial_observation(prompt: str) -> dict[str, Any]:
    """Normalize a user prompt into the observation structure used across Chimera."""
    return {"source": "user", "data": {"text_data": prompt}}


def create_tool_registry(*, allow_file_system: bool = False) -> ToolRegistry:
    """Create the default standalone tool registry."""
    registry = ToolRegistry()
    registry.register_tool(WebSearchTool())
    if allow_file_system:
        registry.register_tool(FileSystemTool())
    return registry


def create_agent(
    *,
    mode: str,
    db_path: str,
    tool_registry: ToolRegistry | None = None,
    cognitive_core: Any | None = None,
) -> Agent | ConsciousnessAwareAgent:
    """Create a configured Chimera agent instance."""
    tool_registry = tool_registry or create_tool_registry()
    if cognitive_core is None:
        from .cognitive_core.prometheus_core import PrometheusCognitiveCore

        cognitive_core = PrometheusCognitiveCore()

    agent_cls = ConsciousnessAwareAgent if mode == "consciousness" else Agent
    return agent_cls(
        cognitive_core=cognitive_core,
        tool_registry=tool_registry,
        db_path=db_path,
    )


def run_agent_turn(
    agent: Agent | ConsciousnessAwareAgent,
    prompt: str,
) -> dict[str, Any]:
    """Run a single observe-think-act cycle and persist the resulting experience."""
    observation = build_initial_observation(prompt)
    agent.working_memory.add(observation)

    action = agent._think(observation)
    agent.working_memory.add(action)

    outcome = agent._act(action)
    agent.working_memory.add(outcome)

    agent.episodic_memory.remember(
        Experience(observation=observation, action=action, outcome=outcome)
    )

    return {
        "observation": observation,
        "action": action,
        "outcome": outcome,
    }


def _print_turn(turn: dict[str, Any], *, as_json: bool) -> None:
    """Render a completed turn for either humans or scripts."""
    if as_json:
        print(json.dumps(turn, indent=2))
        return

    action = turn["action"]
    outcome = turn["outcome"]
    print(f"Action: {action.get('tool_name', 'unknown')}")
    print(json.dumps(action.get("arguments", {}), indent=2))
    print("Outcome:")
    print(json.dumps(outcome, indent=2))


def _run_interactive_shell(
    agent: Agent | ConsciousnessAwareAgent,
    *,
    as_json: bool,
) -> int:
    """Run the lightweight interactive shell."""
    print("Chimera interactive shell started. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("chimera> ").strip()
        except EOFError:
            print()
            return 0

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            return 0

        turn = run_agent_turn(agent, prompt)
        _print_turn(turn, as_json=as_json)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint used by `python -m chimera.main`."""
    parser = build_parser()
    args = parser.parse_args(argv)

    agent = create_agent(
        mode=args.mode,
        db_path=str(Path(args.db_path).expanduser()),
        tool_registry=create_tool_registry(allow_file_system=args.allow_file_system),
    )

    if args.prompt:
        turn = run_agent_turn(agent, args.prompt)
        _print_turn(turn, as_json=args.json)
        return 0

    return _run_interactive_shell(agent, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
