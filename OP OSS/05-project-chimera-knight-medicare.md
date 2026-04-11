# Project Chimera and Knight Medicare

## Key Insight

Knight Medicare does not primarily use Chimera as a generic standalone agent.

It mainly uses Chimera as:

- a memory core
- a consciousness / self-modeling core
- a vendored library behind `chimera-bridge`

That means Chimera’s next job is not “stuff more product-specific therapy logic into core.”

Its next job is:

- become easier to run standalone
- become easier to embed safely
- expose better adapter seams

## What Was Learned From Knight Medicare

The important downstream pattern was:

- KM has its own therapy orchestration path
- KM uses Chimera memory and Narcissus-style state tracking
- KM wraps Chimera behind a FastAPI bridge

So the clean package split should be:

- Chimera owns core cognition, memory, consciousness, CLI, adapters, providers, reusable tools
- Knight Medicare owns therapy product prompts, clinical workflows, patient UX, web APIs, and app-specific guardrails

## Current Public PRs

- Policy-aware tool execution:
  - `#22` https://github.com/LarytheLord/Project-Chimera/pull/22
  - branch: `codex/policy-aware-tool-executor`
  - state on snapshot date: `OPEN`

- Standalone CLI + roadmap:
  - `#23` https://github.com/LarytheLord/Project-Chimera/pull/23
  - branch: `codex/standalone-cli-roadmap`
  - state on snapshot date: `OPEN`

## What PR #22 Did

- added policy-aware tool execution
- added restricted tool behavior
- improved tool exposure boundaries
- added tests around blocked tools and execution behavior

## What PR #23 Did

- added a real standalone `python -m chimera.main` CLI
- added CLI tests
- updated `agi-project` docs
- added a Chimera / Knight Medicare integration roadmap

## Recommended Chimera Plan

### Phase 1

- standalone parity
- stable CLI entrypoint
- light demo flow
- opt-in risky tools only

### Phase 2

- stable adapters and providers
- reusable single-turn execution surfaces
- better downstream integration boundaries

### Phase 3

- provenance and safety hooks
- tool-level policy
- run metadata
- critique / reflection surfaces

### Phase 4

- local emotion models
- constitutional checks
- richer reflection loops
- local fallback LLMs
- multi-agent patterns

## Known Caveat

Earlier validation work intentionally avoided Poetry-wide commands because `agi-project/pyproject.toml` had pre-existing merge-conflict markers on the branch base at the time of implementation. Future work should check whether that is still unresolved before trying to standardize build/test flows.

