# Open Paws Org Map

## High-Level Direction

The best clean-room use of the leak patterns in Open Paws was never “build a Claude clone.”

It was:

- build safer agent/runtime infrastructure
- connect scanner findings to Open Paws quest workflows
- create documented rollout patterns across the org

## Most Important Open Paws Work

### open-paws-platform

PR:

- `#42` https://github.com/Open-Paws/open-paws-platform/pull/42

State:

- `MERGED`

What it did:

- added scanner-to-guild quest ingestion
- converted scanner findings into draft quest proposals
- deduped on scanner finding source IDs
- wrote provenance into guild quest source records

### project-compassionate-code

PR:

- `#13` https://github.com/Open-Paws/project-compassionate-code/pull/13

State:

- `MERGED`

What it did:

- added Open Paws export payload
- later added scan-run provenance metadata

### documentation

PR:

- `#7` https://github.com/Open-Paws/documentation/pull/7

State:

- `MERGED`

What it did:

- documented the clean-room agent/runtime plan across repos

### open-paws-strategy

PR:

- `#1` https://github.com/Open-Paws/open-paws-strategy/pull/1

State:

- `CLOSED`

What it was for:

- recording a clean-room rollout path

### Open-Paws-Tools-Platform

PR:

- `#1` https://github.com/LarytheLord/Open-Paws-Tools-Platform/pull/1

State:

- `OPEN`

What it does:

- adds tool registry sensitivity metadata
- acts as a richer sandbox for future tool UX

## Repos Explicitly Left Alone

Earlier user instructions explicitly ruled out working in:

- `Open-Paws/open-paws-api-gateway`
- `LarytheLord/AFA--open-paws--Resource-Chatbot`

That exclusion should be preserved unless re-approved.

## Organizational Strategy

The best reusable Open Paws pattern is:

- scanner / source system
- export / normalization layer
- runtime / task orchestration
- quest / task destination
- documentation / rollout policy

This is the clean-room infrastructure story that ties the repos together.

