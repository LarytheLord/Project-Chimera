# Project Chimera

A modular cognitive architecture for building self-aware AI agents. Chimera combines episodic memory, metacognitive self-reflection, tool use, and reinforcement learning into a unified framework — all running on free-tier APIs and local CPU models.

```
         ┌─────────────────────────────────────────────────┐
         │              Project Chimera                     │
         │                                                  │
         │   ┌───────────┐  ┌────────────┐  ┌──────────┐  │
         │   │ Prometheus │  │  Narcissus  │  │   RLHF   │  │
         │   │ Cognitive  │  │ Self-Model  │  │  Oracle   │  │
         │   │   Core     │  │ + Metacog   │  │          │  │
         │   └─────┬──────┘  └──────┬──────┘  └────┬─────┘  │
         │         │                │               │        │
         │   ┌─────▼────────────────▼───────────────▼─────┐  │
         │   │            Agent Loop                       │  │
         │   │   perceive → think → act → remember → ...   │  │
         │   └─────┬───────────────────────────────┬──────┘  │
         │         │                               │         │
         │   ┌─────▼──────┐              ┌─────────▼──────┐  │
         │   │   Memory    │              │  Tool Registry  │  │
         │   │ ┌─────────┐ │              │ ┌────────────┐ │  │
         │   │ │ Working  │ │              │ │ Web Search │ │  │
         │   │ │ (deque)  │ │              │ │ File I/O   │ │  │
         │   │ ├─────────┤ │              │ │ Therapy*   │ │  │
         │   │ │ Episodic │ │              │ │ Reflection │ │  │
         │   │ │ (LanceDB)│ │              │ └────────────┘ │  │
         │   │ └─────────┘ │              └────────────────┘  │
         │   └─────────────┘                                  │
         └─────────────────────────────────────────────────────┘
                              * therapy tools when used via Knight Medicare
```

## What makes Chimera different

| Feature | How it works | Cost |
|---------|-------------|------|
| **Episodic Memory** | LanceDB vector store + SentenceTransformers (`all-MiniLM-L6-v2`). Stores experiences, recalls by semantic similarity. | $0 (local disk) |
| **Self-Awareness** | Narcissus system: tracks cognitive states, detects biases, identifies stuck patterns via metacognitive observer. | $0 (in-memory) |
| **Consciousness Simulation** | Self-modeling engine builds a model of its own attention, confidence, and decision patterns over time. | $0 (in-memory) |
| **RLHF** | Reward model (distilbert) scores candidate responses. Oracle selects the best. Trains on preference data. | $0 (CPU training) |
| **LLM Backend** | Gemini 1.5 Flash via API (1,500 req/day free tier). Async + sync support. | $0 (free tier) |
| **Tool Use** | Extensible tool registry with JSON schemas. Built-in: web search, file system. Pluggable: therapy tools, custom tools. | $0 |

**Total infrastructure cost: $0**

## Architecture

```
src/chimera/
├── cognitive_core/          # "Prometheus" — LLM abstraction layer
│   ├── interfaces.py        #   CognitiveCore ABC
│   ├── prometheus_core.py   #   Gemini API implementation
│   ├── model.py             #   Local model architecture (JAX/Flax, future)
│   └── data_loader.py       #   Data preprocessing (future)
│
├── agent/                   # "Janus" — perceive→think→act loop
│   ├── agent.py             #   Agent class (main orchestrator)
│   ├── memory.py            #   VectorEpisodicMemory + WorkingMemory
│   └── tool_user.py         #   Tool ABC, ToolRegistry, WebSearchTool, FileSystemTool
│
├── consciousness/           # "Narcissus" — self-modeling & metacognition
│   ├── narcissus_core.py    #   SelfModelingEngine, MetacognitiveObserver, SelfSimulationFramework
│   ├── integration.py       #   ConsciousnessIntegration (bridge to agent)
│   └── conscious_agent.py   #   ConsciousnessAwareAgent (Agent + Narcissus combined)
│
└── rlhf/                    # Reinforcement Learning from Human Feedback
    ├── reward_model.py      #   RewardModel (distilbert fine-tuning via TRL)
    └── oracle.py            #   RLHFOracle (scores + selects best response)
```

## Quick Start

### Prerequisites
- Python 3.11+
- A Gemini API key (free: [aistudio.google.com](https://aistudio.google.com/app/apikey))

### Install

```bash
# Clone
git clone https://github.com/LarytheLord/Project-Chimera.git
cd Project-Chimera/agi-project

# Install dependencies (pick one)
pip install -r requirements-submodule.txt     # lightweight, no RLHF
poetry install                                 # full install with RLHF + JAX

# Set your API key
export CHIMERA_LLM_API_KEY="your_gemini_api_key"
```

### Run the agent

```python
from chimera.cognitive_core.prometheus_core import PrometheusCognitiveCore
from chimera.agent.agent import Agent
from chimera.agent.tool_user import ToolRegistry, WebSearchTool, FileSystemTool

# Initialize
core = PrometheusCognitiveCore()
tools = ToolRegistry()
tools.register_tool(WebSearchTool())
tools.register_tool(FileSystemTool())

agent = Agent(cognitive_core=core, tool_registry=tools, db_path="./chimera_db")

# Run — agent will perceive, think, act, and remember in a loop
agent.run_main_loop({"task": "Research the latest developments in cognitive architectures"})
```

### Run with consciousness (self-aware agent)

```python
from chimera.consciousness.conscious_agent import ConsciousnessAwareAgent

agent = ConsciousnessAwareAgent(
    cognitive_core=core,
    tool_registry=tools,
    db_path="./chimera_db",
)

# Enable self-reflection
agent.enable_self_reflection()

# The agent now tracks its own cognitive states, detects biases,
# and uses metacognitive insights to improve decision-making
agent.run_main_loop({"task": "Solve a complex problem while monitoring your own reasoning"})

# Inspect the agent's self-model
print(agent.get_self_model())
```

### Run with RLHF (preference-guided responses)

```python
from chimera.rlhf.oracle import RLHFOracle

# Train a reward model first (see scripts/train_reward_model.py)
oracle = RLHFOracle(model_path="./reward_model")

agent = ConsciousnessAwareAgent(
    cognitive_core=core,
    tool_registry=tools,
    db_path="./chimera_db",
    rlhf_oracle=oracle,
    num_candidates=3,  # generate 3 candidates, oracle picks the best
)
```

## Core Components

### Prometheus (Cognitive Core)

The LLM abstraction layer. Currently wraps Gemini 1.5 Flash via HTTP API. Implements the `CognitiveCore` ABC so you can swap in any LLM backend.

```python
from chimera.cognitive_core.prometheus_core import PrometheusCognitiveCore

core = PrometheusCognitiveCore()  # reads CHIMERA_LLM_API_KEY from env
response = core.generate_response({"text_data": "What is consciousness?"})
```

### Janus (Agent Framework)

Perceive → Think → Act loop with vector memory and tool use.

**Memory:**
- `WorkingMemory` — bounded deque (last 20 items), fast in-memory context
- `VectorEpisodicMemory` — LanceDB vector store with SentenceTransformer embeddings. Stores `Experience(observation, action, outcome)` tuples. Recalls by semantic similarity.

```python
from chimera.agent.memory import VectorEpisodicMemory, WorkingMemory, Experience

memory = VectorEpisodicMemory(db_path="./my_db")
memory.remember(Experience(
    observation={"input": "user question"},
    action={"tool": "web_search", "query": "..."},
    outcome={"result": "..."}
))

# Semantic recall
relevant = memory.recall("similar question", top_k=5)
```

**Tools:**
- `Tool` ABC with `name`, `description`, `get_schema()`, `__call__()`
- `ToolRegistry` manages tools, generates JSON schemas for the LLM
- Built-in: `WebSearchTool` (DuckDuckGo + scraping), `FileSystemTool` (read + list)

```python
from chimera.agent.tool_user import Tool

class MyTool(Tool):
    @property
    def name(self): return "my_tool"
    @property
    def description(self): return "Does something useful"
    def get_schema(self): return {"type": "object", "properties": {"input": {"type": "string"}}}
    def __call__(self, input: str): return f"Result for {input}"
```

### Narcissus (Consciousness System)

Self-modeling, metacognition, and cognitive state tracking.

**Components:**
- `SelfModelingEngine` — tracks attention patterns, capability assessments, bias identification
- `MetacognitiveObserver` — analyzes thought processes, detects biases (e.g., confirmation bias from repeated decisions), suggests optimizations
- `SelfSimulationFramework` — simulates proposed cognitive changes before applying them
- `NarcissusConsciousnessCore` — orchestrates all three, records cognitive states

```python
from chimera.consciousness.narcissus_core import NarcissusConsciousnessCore, CognitiveState

narcissus = NarcissusConsciousnessCore(
    cognitive_core=core,
    memory_db_path="./narcissus_db"
)

# Record a cognitive state
state = narcissus.record_cognitive_state(
    thought_process="Analyzing user's emotional state",
    attention_weights={"emotion": 0.6, "context": 0.3, "history": 0.1},
    decision_path=["assess_mood", "select_intervention"],
    confidence=0.75,
    emotional_state={"empathy": 0.8, "concern": 0.6},
    memory_context=["previous sessions"],
    processing_load=0.5,
)

# Introspect
insights = narcissus.perform_introspective_analysis()
# → {self_model_snapshot, metacognitive_insights, suggested_improvements, self_awareness_metrics}
```

### RLHF (Reinforcement Learning from Human Feedback)

Train a reward model on human preferences, then use it to select better responses.

```bash
# 1. Collect preferences
python scripts/collect_preferences.py

# 2. Train reward model
python scripts/train_reward_model.py

# 3. Use in agent (see "Run with RLHF" above)
```

## Integration: Knight Medicare

Chimera powers the AI therapy backend for [Knight Medicare](https://github.com/LarytheLord/knight-medicare), a mental healthcare platform.

```
Patient → Knight Medicare (Next.js)
            → POST /api/therapy
            → chimera-bridge (FastAPI, port 8100)
                → TherapyAgent.process_message()
                    1. PERCEIVE  — WorkingMemory + VectorEpisodicMemory recall
                    2. PATTERNS  — Narcissus MetacognitiveObserver
                    3. ASSESS    — Gemini classifies mood, selects tool
                    4. TOOL      — therapy tool (CBT, journaling, breathing, safety plan)
                    5. RESPOND   — Gemini generates therapeutic response
                    6. RECORD    — LanceDB store + Narcissus CognitiveState
```

The `chimera-bridge/` FastAPI service lives in the KM repo and wraps Chimera's modules for therapy-specific use. Chimera itself remains a general-purpose cognitive architecture.

See [KM Discussion #33](https://github.com/LarytheLord/knight-medicare/discussions/33) for integration details.

## Roadmap

### Completed
- [x] Prometheus cognitive core (Gemini API)
- [x] Agent perceive→think→act loop
- [x] Episodic memory (LanceDB + SentenceTransformers)
- [x] Working memory (bounded deque)
- [x] Tool registry + web search + file system tools
- [x] Narcissus consciousness system (self-modeling, metacognition, simulation)
- [x] Consciousness-aware agent
- [x] RLHF reward model + oracle
- [x] Knight Medicare therapy integration

### In Progress
- [ ] `chimera/__init__.py` fix for submodule compatibility (#9)
- [ ] Standalone CLI entry point (#15)
- [ ] Local emotion detection via HuggingFace (#16)
- [ ] Reflexion self-critique + Constitutional AI guardrails (#17)

### Planned
- [ ] Local LLM fallback — SmolLM2 GGUF on CPU (#18)
- [ ] ACT-R memory decay + temporal validity (#20)
- [ ] Feed consciousness insights back into prompts (#21)
- [ ] Three-tier memory (semantic + episodic + procedural)
- [ ] DSPy prompt optimization
- [ ] Therapist RLHF feedback loop

See [Discussion #19](https://github.com/LarytheLord/Project-Chimera/discussions/19) for the full evolution roadmap.

## Development

```bash
# Run tests
cd agi-project
poetry run pytest tests/

# Verify core imports
python -c "
from chimera.agent.memory import VectorEpisodicMemory, WorkingMemory
from chimera.consciousness.narcissus_core import NarcissusConsciousnessCore
from chimera.cognitive_core.prometheus_core import PrometheusCognitiveCore
print('All imports OK')
"
```

### Submodule usage (for Knight Medicare)
```bash
# In the KM repo:
git submodule update --init lib/chimera
cd lib/chimera && git checkout v0.2.0-km-ready
```

### Contributing
1. Create a branch off `master`
2. Make changes in `agi-project/src/chimera/`
3. Run tests: `poetry run pytest`
4. Tag a release: `git tag v0.x.x-km-ready`
5. Push: `git push origin master --tags`

## Team

- **Abid (LarytheLord)** — Architecture, KM integration, project lead
- **Prit (Prit-P2)** — Chimera core modules, Python specialist

## License

MIT
