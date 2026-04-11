# Leak-Derived Patterns

These are the architectural ideas worth reusing in a clean-room way.

## Core Patterns

- Typed tool registry
  One central definition of tools, schemas, permissions, and metadata.
- Planner / executor split
  Separate “reason about what to do” from “execute constrained actions.”
- Background task / session model
  Long-running work should have explicit status, logs, progress, and resumability.
- Provenance and audit trails
  Record what ran, with what inputs, what outputs, and what failed.
- Memory / session compaction
  Keep long-running work usable by compressing context over time.
- Policy-aware execution
  Sensitive actions should not be implicitly available just because they exist.
- Feature flags and controlled rollout
  Gate riskier capabilities behind explicit switches.
- Connector abstraction
  External systems should be wrapped behind stable interfaces.

## Safe Translation Into Our Repos

- `Pocket Quant`
  Use task-run orchestration, sync provenance, and background job state.
- `Project Chimera`
  Use policy-aware tool execution, standalone CLI surfaces, and adapter boundaries.
- `Project Compassionate Code`
  Use scanner-run provenance, repo export schemas, and orchestrated PR workflows.
- `Open Paws`
  Use scanner ingestion, quest/task orchestration, runtime policy, and documented rollouts.

## What Not To Reuse

- Specific leaked implementation details.
- Internal names or comments from the leak.
- Proprietary prompt phrasing.
- Any branding or “clone” framing.

## The Real Lesson

The value is not “the leaked code.”

The value is understanding what makes an agentic product feel real:

- explicit runtime state
- permission boundaries
- progress visibility
- modular connectors
- durable sessions
- strong operator ergonomics

