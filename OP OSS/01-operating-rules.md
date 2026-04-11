# Operating Rules

These are the working rules that should be treated as persistent project context.

## Git / Branch Discipline

- Never commit directly on `main` or `master`.
- Always create or continue work on a non-default branch.
- Always open a PR back to the repo's default branch.
- Default to a draft PR unless there is a reason not to.
- Never merge PRs directly from the agent side.

## Clean-Room Rule

- Do not copy leaked files, prompts, names, comments, or implementations into shipping repos.
- Use the leak only as a reference architecture.
- Reimplement ideas from scratch inside our own repos.

## Repos To Avoid Unless Re-Approved

These repos were explicitly excluded during earlier work:

- `Open-Paws/open-paws-api-gateway`
- `LarytheLord/AFA--open-paws--Resource-Chatbot`

Unless the user explicitly changes the instruction, do not work in those repos.

## Practical Working Rules

- Treat existing uncommitted work as user-owned unless clearly created by the current change.
- In dirty repos, scope commits intentionally instead of sweeping everything in.
- Prefer small slices that can be reviewed and shipped independently.
- For local-only repos with no remote, create a clean branch and commit there, but do not assume a remote exists.

## Product / Strategy Rules

- Don’t sell the leak.
- Sell the product or workflow the clean-room implementation unlocks.
- Avoid generic “AI agent” positioning when a vertical product story exists.
- Prioritize the products with the clearest monetization path:
  - `Pocket Quant`
  - `Knight Medicare` + `Project Chimera`
  - `Project Compassionate Code`

## Current Known Caveats

- Some old `/tmp` working clones used during implementation are no longer present.
- GitHub PRs are now the most reliable record for several past changes.
- `Pocket Quant` is still locally dirty beyond the PR-scoped runtime work; do not assume PR `#1` contains all local app scaffolding.

