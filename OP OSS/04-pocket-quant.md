# Pocket Quant

## Best Product Positioning

`Pocket Quant` should be positioned as:

`The AI quant copilot for Indian crypto investors`

But the actual product should behave more like:

`a portfolio intelligence layer: sync + risk + digest + alerts`

not a generic chatbot.

## Why This Can Stand Out

Pocket Quant should not compete head-on on:

- “all assets in one place”
- “most integrations”
- “best place to trade”
- generic “AI insights”

Those lanes are already crowded by products like `Delta`, `CoinStats`, `CoinMarketCap Portfolio`, `Kubera`, `KoinX`, `CoinSwitch`, and `CoinDCX`.

The cleanest wedge is:

- India-first
- INR-native
- read-only
- portfolio-aware
- risk and explanation oriented
- built for holders/investors rather than pure traders

## Unique Angle

The uniqueness is not “AI.”

The uniqueness is:

- explain my actual portfolio
- tell me what changed and why it matters
- show me concentration and risk before it becomes a problem
- do it in INR and in a lightweight workflow

## Website vs Mobile

Start web-first.

Recommended release shape:

- Phase 1: responsive website
- Phase 2: PWA
- Phase 3: native mobile only if usage and paid retention justify it

Reason:

- the codebase is already Next.js web
- charts, settings, and exchange connection flows are easier to ship on web first
- mobile-quality UX still matters, but app-store friction does not help early validation

## What Was Implemented

Public repo:

- https://github.com/LarytheLord/pocket-quant

Current PR:

- `#1` https://github.com/LarytheLord/pocket-quant/pull/1

Included work:

- runtime task policies
- task run state model
- task-run tests
- portfolio sync response normalization
- live sync route emits task-run state
- live sync route persists daily snapshot
- store tracks portfolio mode, portfolio data, task runs, errors
- dashboard, header, digest, and risk pages now reflect runtime state

Core idea:

- portfolio sync becomes the first real background-style workflow
- digest/risk pages now acknowledge prerequisites instead of staying disconnected placeholders

## Current Local Caveat

The local repo is still dirty beyond PR `#1`.

There are many other local files and scaffolded app pieces not included in the PR yet. Future work should continue with scoped commits and new PRs rather than force-pushing everything into the current one.

## Recommended Next Steps

1. Finish the baseline app shell and auth flows outside PR `#1`.
2. Add actual digest generation as a tracked `ai_digest` workflow.
3. Add real risk computation as a tracked `risk_snapshot` workflow.
4. Add alerts delivery and history as a tracked `alert_evaluation` workflow.
5. Add Telegram / email delivery for digest and alerts.

## MVP Release

Strong v1:

- CoinDCX sync
- dashboard in INR
- daily snapshot persistence
- daily digest
- risk score
- concentration / volatility alerts
- responsive web

Avoid at launch:

- DeFi sprawl
- tax suite
- NFT tracking
- AI buy/sell signals
- broad prediction claims

## Monetization

Suggested pricing direction:

- free: 1 exchange, dashboard, manual sync, basic alerts
- paid: daily digest, richer risk analytics, Telegram alerts, deeper insights

Suggested India-first price band:

- `₹499–999/month`
- `₹4,999–7,999/year`

