# Chimera and Knight Medicare

This roadmap turns the current Knight Medicare integration into a cleaner package contract for `chimera`.

## What Knight Medicare Uses Today

Knight Medicare vendors Chimera as a library and mostly relies on three subsystems:

- `chimera.agent.memory` for `VectorEpisodicMemory` and `WorkingMemory`
- `chimera.consciousness` for `NarcissusConsciousnessCore` and `ConsciousnessIntegration`
- the package structure itself, via `chimera-bridge`, which wraps Chimera behind FastAPI routes

Knight Medicare does **not** currently use the standalone `Agent` or `ConsciousnessAwareAgent` loop for therapy. Its `chimera-bridge/adapters/km_agent.py` runs a separate single-turn therapy pipeline with its own prompts, async provider, and therapy-tool registry.

That means Chimera's next job is not "move all Knight Medicare code into core overnight." The next job is to become easier to run standalone and easier to embed safely.

## Package Priorities

### Phase 1: Standalone Parity

Ship the minimum surfaces needed for local usage and demos:

- a supported `python -m chimera.main` entrypoint
- a reusable single-turn runner that mirrors the observe-think-act cycle
- opt-in local file-system access rather than exposing it by default
- docs that explain how standalone mode differs from Knight Medicare mode

This PR covers that phase.

### Phase 2: Stable Adapter Seams

Create explicit extension points so downstream apps stop importing deep internals ad hoc:

- `chimera.adapters.therapy` for Knight Medicare's therapy loop
- `chimera.providers` for sync and async LLM backends
- `chimera.tools` for domain-specific tool packs such as therapy interventions
- a shared response shape for single-turn runs, tool outcomes, and memory writes

The goal is to let Knight Medicare depend on supported interfaces instead of carrying a parallel orchestration layer forever.

### Phase 3: Provenance and Safety

Bring clean-room lessons from modern agent runtimes into Chimera itself:

- turn-level run metadata for prompts, tool calls, and outcomes
- critique hooks before returning high-risk responses
- guardrails for tool exposure and domain-sensitive behavior
- policy-aware configuration for future restricted capabilities

This matters more for Knight Medicare than for the standalone demo because therapy flows need traceability.

### Phase 4: Research Features After the Core Is Stable

After the package boundaries are solid, Chimera can absorb higher-order work like:

- local emotion models
- constitutional checks
- multi-agent orchestration
- local fallback LLM providers
- richer reflection loops

Knight Medicare's own strategy docs are directionally right here: the foundation should come before extra AGI breadth.

## Recommended Ownership Split

- Chimera repo owns core cognition, memory, consciousness, CLI, adapters, providers, and reusable tools.
- Knight Medicare owns product prompts, patient workflows, clinician UX, web APIs, and app-specific safety policies.

That split keeps Chimera reusable while letting Knight Medicare iterate on therapy behavior quickly.
