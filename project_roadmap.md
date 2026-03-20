# Project Chimera — Roadmap

## Phase 1: Foundational Intelligence — COMPLETED

| Component | Status | Description |
|-----------|--------|-------------|
| Prometheus Core | Done | Gemini 1.5 Flash API integration via `PrometheusCognitiveCore` |
| Episodic Memory | Done | `VectorEpisodicMemory` — LanceDB + SentenceTransformers (`all-MiniLM-L6-v2`) |
| Working Memory | Done | Bounded deque (20 items), fast in-memory context |
| Agent Loop | Done | `Agent` class — perceive→think→act with tool use and memory |
| Tool Registry | Done | `Tool` ABC + `ToolRegistry` + WebSearch + FileSystem tools |
| RLHF | Done | `RewardModel` (distilbert) + `RLHFOracle` for preference-guided selection |
| Narcissus | Done | Self-modeling, metacognitive observer, cognitive state tracking |
| Consciousness-Aware Agent | Done | `ConsciousnessAwareAgent` — Agent + Narcissus integrated |
| Knight Medicare Integration | Done | Therapy backend via `chimera-bridge/` (FastAPI) |

## Phase 2: Enhanced Cognition — IN PROGRESS

| # | Feature | Status | Priority |
|---|---------|--------|----------|
| #9 | Fix `chimera/__init__.py` for submodule compat | Blocker | P0 |
| #15 | Standalone CLI + dual-mode package restructure | Open | P1 |
| #16 | Local emotion detection (HF distilroberta, 7 labels, CPU) | Open | P1 |
| #21 | Feed Narcissus consciousness insights back into prompts | Open | P1 |
| #17 | Reflexion self-critique + Constitutional AI guardrails | Open | P2 |
| #18 | Local LLM fallback (SmolLM2-360M GGUF, CPU) | Open | P2 |
| #20 | ACT-R memory decay + temporal memory validity | Open | P2 |

**Priority order:** #9 → #15 → #16 → #21 → #17 → #20 → #18

## Phase 3: Advanced Research — PLANNED

| Feature | Description |
|---------|-------------|
| Three-tier memory | Semantic (patient facts) + episodic (sessions) + procedural (strategies) |
| "Dreaming" | Async background synthesis of patient/user representations |
| Global Workspace | Components compete for attention; crisis signals always win |
| Cognitive state machine | SOAR/ACT-R-inspired: PERCEIVE→ATTEND→RETRIEVE→DELIBERATE→REFLECT→ACT→LEARN |
| DSPy prompt optimization | Auto-tune Gemini prompts via bootstrapping |
| RLHF with therapist feedback | Psychologist rates AI therapy responses → preference pairs → better model |
| Introspective compression | Compress cognitive states into stable personality traits |
| Multi-agent inner council | Assessment / Intervention / Empathy / Safety sub-agents |

## Constraints

- **$0 budget** — free-tier APIs, local models, open-source only
- **CPU only** — no GPU requirements
- **Dual-mode** — every feature must work standalone AND as a KM submodule
- **Therapy-safe** — never break patient-facing therapy quality for research features

## References

- [Evolution Plan](https://github.com/LarytheLord/knight-medicare/blob/feat/chimera-integration/docs/CHIMERA-EVOLUTION-PLAN.md) — full architecture plan
- [Discussion #19](https://github.com/LarytheLord/Project-Chimera/discussions/19) — enhancement roadmap
- [KM Discussion #33](https://github.com/LarytheLord/knight-medicare/discussions/33) — therapy integration status
