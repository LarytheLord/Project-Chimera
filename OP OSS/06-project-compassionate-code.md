# Project Compassionate Code

## Strongest Codex Hackathon Candidate

If choosing one past project to pitch for an OpenAI Codex hackathon, this is the best one.

Reason:

- it is already naturally a Codex-native workflow
- it reads repositories
- it finds concrete improvement opportunities
- it can generate contribution artifacts
- it can orchestrate PR preparation

That maps directly onto what Codex is good at.

## Best Pitch Framing

Do not lead with activism.

Lead with:

`A Codex-powered open source contribution agent that finds high-merge-probability improvements, drafts the fix, and prepares a review-ready PR.`

That pitch is cleaner, broader, and much stronger in a hackathon setting.

## Current Public Status

PR:

- `#13` https://github.com/Open-Paws/project-compassionate-code/pull/13

State on snapshot date:

- `MERGED`

What landed:

- Open Paws export format
- stable finding IDs
- later scan-run provenance metadata

## Why It Is Such A Good Demo

A powerful live demo looks like:

1. paste a repo URL
2. scanner finds opportunities
3. Codex selects a legitimate one
4. Codex drafts or applies the change
5. PR summary is generated
6. optionally open a draft PR

That is a much stronger hackathon demo than a generic agent chat.

## Possible Product Shapes

- GitHub App for contribution intelligence
- internal OSS team assistant
- issue / PR triage and suggestion tooling
- campaign / cause-driven contribution infrastructure

## Best Immediate Hackathon Scope

Build a lightweight UI around the existing scanner + orchestrator:

- repo input
- ranked opportunity list
- “generate patch”
- “draft PR summary”
- “open draft PR”

## Why It Beats Pocket Quant For The Hackathon

`Pocket Quant` may be the better business wedge.

But `Project Compassionate Code` is the better Codex hackathon wedge because the relationship to code analysis and code change is direct and obvious.

