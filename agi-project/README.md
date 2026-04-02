# Chimera AGI — Source Package

This directory contains the core Python package (`src/chimera/`) and all supporting infrastructure.

> For the full project overview, architecture diagrams, and quickstart guide, see the [root README](../README.md).

## Directory Layout

```
agi-project/
├── src/chimera/                 # The core package (importable as `chimera.*`)
│   ├── main.py                  #   Standalone CLI entry point (`python -m chimera.main`)
│   ├── cognitive_core/          # Prometheus — LLM abstraction
│   │   ├── interfaces.py        #   CognitiveCore ABC (implement this for new LLM backends)
│   │   ├── prometheus_core.py   #   Gemini 1.5 Flash implementation
│   │   ├── model.py             #   JAX/Flax local model (placeholder)
│   │   └── data_loader.py       #   Data preprocessing (placeholder)
│   │
│   ├── agent/                   # Janus — the agent framework
│   │   ├── agent.py             #   Agent: perceive→think→act loop
│   │   ├── memory.py            #   VectorEpisodicMemory (LanceDB), WorkingMemory (deque)
│   │   └── tool_user.py         #   Tool ABC, ToolRegistry, WebSearchTool, FileSystemTool
│   │
│   ├── consciousness/           # Narcissus — self-awareness & metacognition
│   │   ├── narcissus_core.py    #   CognitiveState, SelfModelingEngine, MetacognitiveObserver
│   │   ├── integration.py       #   ConsciousnessIntegration (agent-friendly wrapper)
│   │   └── conscious_agent.py   #   ConsciousnessAwareAgent (Agent + Narcissus)
│   │
│   └── rlhf/                    # Reinforcement Learning from Human Feedback
│       ├── reward_model.py      #   RewardModel (distilbert, trains on CPU)
│       └── oracle.py            #   RLHFOracle (scores candidates, picks best)
│
├── scripts/                     # Utility scripts
│   ├── collect_preferences.py   #   Interactive RLHF preference collection
│   ├── train_reward_model.py    #   Train the reward model on preference data
│   └── verify_data.py           #   Verify data integrity
│
├── tests/                       # Test suite
│   ├── test_agent_logic.py      #   Agent + memory + tool tests
│   ├── test_main.py             #   Standalone CLI tests
│   ├── test_consciousness.py    #   Full Narcissus integration tests
│   └── test_consciousness_simple.py  # Unit tests for consciousness components
│
├── pyproject.toml               # Poetry config (agi-chimera v0.2.0)
├── requirements-submodule.txt   # Lightweight deps (no RLHF/JAX)
└── Dockerfile                   # Container build
```

## Setup

### Option A: Lightweight (no RLHF, no JAX)
```bash
pip install -r requirements-submodule.txt
export CHIMERA_LLM_API_KEY="your_gemini_key"
```

### Option B: Full install (RLHF + JAX + everything)
```bash
poetry install
poetry shell
export CHIMERA_LLM_API_KEY="your_gemini_key"
```

## Running Tests

```bash
# With poetry
poetry run pytest tests/ -v

# Without poetry (if you used pip install)
PYTHONPATH=src pytest tests/ -v
```

## Running Chimera Standalone

```bash
# Single prompt, standard agent
PYTHONPATH=src python -m chimera.main --prompt "Research the latest open-source vector databases" --json

# Consciousness-aware interactive shell
PYTHONPATH=src python -m chimera.main --mode consciousness

# Opt in to the read-only file system tool
PYTHONPATH=src python -m chimera.main --allow-file-system --prompt "List the files in the current directory"
```

The standalone CLI exposes web search by default. Local file-system access is available only when `--allow-file-system` is passed.

## Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `PrometheusCognitiveCore` | `cognitive_core.prometheus_core` | Gemini API wrapper, implements `CognitiveCore` ABC |
| `Agent` | `agent.agent` | Main perceive→think→act loop |
| `ConsciousnessAwareAgent` | `consciousness.conscious_agent` | Agent + Narcissus self-reflection |
| `VectorEpisodicMemory` | `agent.memory` | LanceDB-backed semantic memory |
| `WorkingMemory` | `agent.memory` | Bounded deque (default 20 items) |
| `ToolRegistry` | `agent.tool_user` | Manages tools, generates JSON schemas for LLM |
| `NarcissusConsciousnessCore` | `consciousness.narcissus_core` | Orchestrates self-modeling + metacognition |
| `ConsciousnessIntegration` | `consciousness.integration` | Agent-friendly wrapper for Narcissus |
| `RewardModel` | `rlhf.reward_model` | Distilbert fine-tuned on preference pairs |
| `RLHFOracle` | `rlhf.oracle` | Scores candidate responses, selects best |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CHIMERA_LLM_API_KEY` | Yes | Gemini API key ([get one free](https://aistudio.google.com/app/apikey)) |
| `GEMINI_API_KEY` | Fallback | Alternative name for the same key |

## Versioning

Releases that are compatible with Knight Medicare are tagged `v0.x.x-km-ready`. The submodule in KM pins to these tags.

| Tag | What changed |
|-----|-------------|
| `v0.2.0-km-ready` | Clean restructure, modules at `src/chimera/*`, shims at old paths, large files removed |
| `v0.3.0-km-ready` | (planned) `chimera/__init__.py` fix |
| `v0.4.0-standalone` | (planned) Standalone CLI + package restructure |

The Knight Medicare integration roadmap now lives in [docs/KNIGHT_MEDICARE_INTEGRATION_PLAN.md](docs/KNIGHT_MEDICARE_INTEGRATION_PLAN.md). It reflects how KM actually consumes Chimera today: as a memory and consciousness core behind a thin FastAPI bridge.
