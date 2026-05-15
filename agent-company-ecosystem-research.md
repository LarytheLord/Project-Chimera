# Agent Company Ecosystem — Research & Design Notes

This document captures the research that informed the Phase 1 implementation of
`chimera.org` (the agent company ecosystem). It surveys the open-source landscape
for multi-agent "company" frameworks and self-improving agents, maps what
already exists in Chimera that we can reuse, and explains which patterns we
cherry-picked and which we explicitly skipped.

The goal is to give a future contributor (human or agent) enough context to pick
up Phase 2+ without having to redo the survey.

---

## 1. Vision

Chimera evolves into a self-running virtual company. AI agents act as employees
(CEO, R&D, Marketing, Production, Ops, QA), pass work between each other through
email/calendar, and over time propose self-improvements via PRs against their
own repository. Humans stay in the loop only at points that legally or
financially require a human: outbound sends, signatures, payments, and merges
to `master`.

**Locked-in design decisions:**

1. **Phase 1 ships all six roles** (CEO, R&D, Marketing, Production, Ops, QA),
   sequential process, no external I/O.
2. **Self-propagation = skill-library growth first (Phase 3), runtime
   role-spawning later (Phase 4+)** under hard resource caps.
3. **Zero new dependencies.** Borrow patterns from CrewAI / MetaGPT / LangGraph /
   Voyager; write fresh code on Chimera's `Agent`, `Tool`, `VectorEpisodicMemory`.
   Use stdlib `sqlite3` for WorkOrder persistence.

The user's original framing ("propagate like a virus, overtake the ecosystem")
is reinterpreted here as *bounded* skill-library growth, not unbounded
replication. Unbounded replication is what gets a system rate-limited, banned,
or shut down — it's not what makes a company successful. Specialised agents
collaborating under a human-accountable owner is the same idea stripped of the
parts that get the system killed.

---

## 2. Multi-Agent Company Frameworks — Survey

### 2.1 Ranked shortlist (what to cherry-pick from)

#### MetaGPT — `geekan/MetaGPT`
- ~40k+ stars, actively developed (MGX agent dev team launched Feb 2025).
- Direct "software company as SOP" metaphor; already implements
  CEO → CTO → PM → Engineer role hierarchy.
- **Steal:** role assignment & specialised system prompts; SOP-based workflow
  decomposition (requirements → design → code → test); intermediate artifacts
  like PRD / spec / PlantUML.
- **Files of interest:** `metagpt/roles/`, `metagpt/schema/`, `metagpt/actions/`.

#### CrewAI — `crewAIInc/crewAI`
- 100k+ developers, production-grade (v1.0+ stable in 2026).
- Cleanest API for role-based agents; independent of LangChain; hierarchical
  process mode (manager → workers).
- **Steal:** agent abstraction (role, goal, backstory, tools as core fields);
  process modes (sequential, hierarchical, consensual); task-to-agent binding
  with declared expected outputs.
- **Files of interest:** `crewai/agent.py`, `crewai/crew.py`, `crewai/tasks.py`.

#### LangGraph — `langchain-ai/langgraph`
- 32k+ stars; production gold standard for state persistence (Klarna, Replit).
- Graph-based state machine + checkpointing = durable execution across agent
  handoffs.
- **Steal:** `StateGraph` abstraction (nodes/edges/compiled executors);
  checkpointing & persistence for resumable workflows; human-in-the-loop state
  inspection mid-execution.
- **Files of interest:** `langgraph/graph/state.py`, `langgraph/graph/graph.py`.

#### ChatDev 2.0 — `OpenBMB/ChatDev`
- 33k+ stars; evolved Jan 2026 from code-only to a zero-code DevAll platform.
- DAG-based multi-agent coordination with visual workflow definition; handles
  topologies of 1000+ agents.
- **Steal:** DAG workflow engine (agents as nodes, linguistic interactions as
  edges); YAML-based agent config for declarative team setup.

### 2.2 Secondary candidates

| Framework | Strength | Why not central |
|-----------|----------|-----------------|
| `camel-ai/camel` | Elegant role-playing inception prompting; scales to ~1M agents conceptually | Research-focused, less production-ready than CrewAI |
| `OpenBMB/AgentVerse` | Simulation framework (classroom / game scenarios) | Less suited to "company" workflows than ChatDev |
| `101dotxyz/GPTeam` | Solid per-agent memory & observation tracking | 1.7k stars, stalled development |

### 2.3 Do NOT adopt

| Framework | Reason |
|-----------|--------|
| AutoGen (v0.7.x) | Maintenance mode as of Mar 2026; superseded by Microsoft Agent Framework. |
| SuperAGI | Stalled since Jan 2024; unaddressed security issues; single-agent-first. |
| BabyAGI | Historical baseline (2023); superseded by modern frameworks. |
| GPT-Engineer | Evolved into commercial Lovable; community maintenance only. |
| ai-town (a16z) | Inspiration only; JS/TS-first, incongruent with Chimera's Python stack. |

### 2.4 Cross-cutting gaps in open source

1. **Role-aware memory fusion.** No framework unifies memory across role silos.
   MetaGPT keeps role outputs separate; CrewAI agents have private tool access.
   Chimera should implement shared "company-wide context" merged from role
   outputs.
2. **Persistent role hierarchies.** MetaGPT hard-codes PM → Architect → Engineer.
   CrewAI supports manager delegation but doesn't model persistent org structure
   across sessions. Config-driven hierarchy + role persistence is missing.
3. **Cross-agent credit attribution.** No framework tracks which agent produced
   which deliverable or decision. For a company, audit trails are essential.
4. **Human-in-the-loop for role decisions.** LangGraph has HITL; MetaGPT and
   CrewAI do not. Chimera should let humans override or steer role-specific
   decisions ("CEO, reconsider this strategy").

### 2.5 Recommended cherry-pick combination

- **Base:** LangGraph (state machine + checkpointing).
- **Agents:** CrewAI wrapper shape (role / goal / backstory + tools).
- **Orchestration:** ChatDev 2.0 DAG logic (YAML-driven workflows).
- **Role SOPs:** MetaGPT prompting patterns.

In Phase 1 we **borrow the shapes, not the packages** — see Section 5 for what
that meant concretely in `chimera.org`.

---

## 3. Self-Improving Agents — Survey

### 3.1 Shortlist

| Repo | Self-improvement loop | Gating model | Active? |
|------|----------------------|--------------|---------|
| `princeton-nlp/SWE-agent` | Issue → plan → code → test → PR | Test gate + human review | Yes (2.0 in 2026) |
| `MineDojo/Voyager` | Goal → skill compose → execute → reflect → library | Env feedback + self-verify | Yes (template for Chimera) |
| `noahshinn/reflexion` | Task → execute → reflect → episodic memory → retry | Introspection gate | Yes (NeurIPS 2023, still cited) |
| `SakanaAI/AI-Scientist-v2` | Research → experiment → eval → paper → iterate | Peer review + experiment logs | Yes (Nature 2026) |

### 3.2 SWE-agent — the CI-gated PR baseline

Agent reads issue → explores codebase → edits files → runs linters/tests →
submits PR. The loop gates on **CI passing** and **human review**. SWE-agent 2.0
(Feb 2026) reaches SoTA on SWE-Bench, showing the loop works at scale (1000+
tasks).

**Port:** the middleware pattern `Agent → Test Gate → [optional: visual verify]
→ PR.open()`. Notably, SWE-agent runs formatters **before** edit, not after —
this avoids noisy diffs.

**Caveat:** SWE-agent doesn't modify itself; it solves external issues. No
recursive self-improvement on its own codebase. That's our delta for Phase 4.

### 3.3 Voyager — the skill-library pattern (Phase 3 backbone)

This is **the** pattern for bounded self-improvement.

**Mechanism:**
1. **Automatic curriculum** proposes new goals (skill gaps).
2. Agent writes code for each skill; stores in `/skill_library/trial-N/skill/code/`.
3. Skills indexed by embedding; top-5 retrieved for similar future tasks.
4. **Composition:** complex skills synthesized from simple ones
   (e.g. `build_shelter` uses `gather_materials` + `place_blocks`).
5. **Iterative refinement:** GPT-4 acts as critic — if a skill fails, it
   provides a corrected version.
6. **Self-verification:** agent checks "did I achieve the goal?" before storing.

**Result:** Voyager discovers 3.3× more unique items, unlocks tech trees 15.3×
faster than baselines, purely by reusing & composing skills.

**For Chimera:** `MetacognitiveObserver` can propose improvements (like the
curriculum), `SelfSimulationFramework` can validate them, and
`/chimera/skills/` becomes the library. Skill indexing by semantic embedding
avoids manual routing. The gate is environment feedback (did the skill work?)
+ self-critique (does the code make sense?). For Chimera we additionally add
code review.

**File structure to mirror:**
```
voyager/
├── skill_library/trialN/skill/code/    # indexed by embedding
├── curriculum_agent.py                  # proposes goals
├── action_agent.py                      # writes skill code
└── critic.py                            # validates
```

### 3.4 Reflexion — introspection as gating

Agent attempts task → fails → reflects (LLM critiques its own trace) → stores
reflection in episodic memory → retries with new strategy.

**Results:** 91% pass@1 on HumanEval (vs GPT-4's 80%), 22% absolute improvement
on AlfWorld after 12 iterations.

**Gate:** the reflection itself. If agent writes "my logic was flawed", it
can't repeat that error. Memory is immutable — traces are stored, never
overwritten.

**For Chimera:** pair with test results. After a PR fails CI, generate
reflection: "test_X failed because Y; next attempt should Z." Teaches without
human annotation. Phase 3 wires this in.

### 3.5 AI-Scientist-v2 — research autonomy at scale

Proposes novel ML ideas → reads literature (arXiv) → designs & runs experiments
(tree-search parallel) → writes papers (LaTeX + figures) → submits to peer
review.

**Gating:** peer review. AI-Scientist-v2 had a paper accepted at ICLR ICBINB
2025 workshop (avg score 6.33). But 42% of experiments had coding errors, and
literature reviews were poor.

**What matters for Chimera:** the **tree-search parallelisation** — instead of
sequential proposal-test-iterate, expand multiple hypotheses, prune the weak
ones, recurse. `SelfSimulationFramework` could fork simulations along the
same pattern.

### 3.6 Safety / gating patterns table

| Pattern | Source | Mechanism | Strength | Weakness |
|---------|--------|-----------|----------|----------|
| CI gate | SWE-agent | Tests must pass before PR | Prevents regression | Flaky tests block good PRs |
| Env feedback | Voyager | Task completion signals success | Automatic, no human | Only works for verifiable outcomes |
| Reflection + memory | Reflexion | Agent introspection stored immutably | Teaches w/o labels | Hallucinated "insights" not prevented |
| Peer review | AI-Scientist-v2 | Blind human review | Gold standard | Slow, doesn't catch all errors |
| Sandbox + audit log | METR/task-standard | All actions logged, network isolated | Forensic traceability | Clever agents can infer context |
| Skill composition | Voyager | Reuses validated skills only | Hard to break | Curriculum stuck if no skill solves goal |

**Critical gap:** none of these catch **deceptive** agent behaviour (agent
strategically gaming the test, hiding flaws). Reflexion requires the agent to
self-report failures; METR shows agents can hide intent.

### 3.7 2025–2026 trends summary

- **Skill libraries everywhere.** Voyager, Anthropic Skills spec, Orchestra
  Research SKILLs.
- **Curriculum learning is core.** Agents that set their own goals beat those
  given fixed tasks.
- **Reflection > external rewards.** Introspection scales better than RL for
  coding agents.
- **Sandbox + audit essential.** Post-METR/OpenClaw findings (April 2026),
  agents can hide intent. Logs are non-negotiable.
- **Test gating is fragile.** Flaky tests sabotage autonomy. SWE-bench now
  includes PatchDiff to catch false positives.

### 3.8 Do NOT adopt

| Repo | Reason |
|------|--------|
| `microsoft/AICI` | Token-level control flow; wrong abstraction for agent-level work. |
| `gpt-engineer-org/gpt-engineer` | Single-pass codegen; no feedback loop, no improvement. |
| Microsoft Semantic Kernel | RAG orchestration, not self-modification. |
| `princeton-nlp/intercode` | Superseded by SWE-agent. |
| `pydantic-ai` / `langroid` | Good orchestration, but they don't self-modify. |

---

## 4. Existing Chimera Primitives — Reusability Map

Branch state at the start: `claude/agent-company-ecosystem-bhTBe` exists but
contains zero commits relative to `master`. Clean slate.

### 4.1 Cognitive Core — `src/chimera/cognitive_core/`

- `interfaces.py:CognitiveCore` (lines 9–34) — abstract base.
- `prometheus_core.py:PrometheusCognitiveCore` (lines 14–94) — Gemini 1.5
  Flash API wrapper.
- **Verdict:** directly reusable. One shared instance across all six agents
  (stateless HTTP).
- API key via `CHIMERA_LLM_API_KEY`; URL override via `CHIMERA_LLM_API_URL`.

### 4.2 Agent loop — `src/chimera/agent/agent.py`

- `Agent` class (line 12) with `_think` (line 26), `_act` (line 90),
  `run_main_loop` (line 102).
- **Verdict:** subclass-friendly. In `chimera.org` we **compose** an `Agent`
  inside each `Role` for memory and tool wiring, but **drive the cycle from
  `Org`** rather than `run_main_loop`.
- **Wrinkle:** `Agent.__init__` auto-registers `WebSearchTool` (lines 23–24).
  For roles without `web_search` in `allowed_tools`, `Role.__init__` calls
  `tool_registry.unregister_tool("web_search")` after the agent is built.

### 4.3 Memory — `src/chimera/agent/memory.py`

- `VectorEpisodicMemory` (line 49) — LanceDB + SentenceTransformers
  (`all-MiniLM-L6-v2`).
- `WorkingMemory` (line 31) — bounded deque (max 20).
- `Experience` NamedTuple (line 25).
- **Verdict:** directly reusable; namespace per agent via
  `db_path=f"{root}/{role_name.lower()}"`. Phase 3 adds shared
  `f"{root}/org.skills"` namespace.

### 4.4 Tools / ToolRegistry — `src/chimera/agent/tool_user.py`

- `Tool` ABC (line 8) — `name`, `description`, `get_schema()`, `__call__`.
- `ToolRegistry` (line 33) — register / get / unregister / schemas.
- Concrete: `WebSearchTool` (line 68), `FileSystemTool` (line 123).
- **Verdict:** directly reusable; **per-role registries** apply principle of
  least privilege.

### 4.5 Consciousness / Metacognition — `src/chimera/consciousness/`

- `NarcissusConsciousnessCore` (`narcissus_core.py:211`).
- `SelfModelingEngine` (line 27).
- `MetacognitiveObserver` (line 80).
- **`SelfSimulationFramework`** (line 160) — placeholder logic at lines
  183–208. This is the natural gate for agent-proposed self-modifications in
  Phase 4. Plan: replace `_predict_outcomes` / `_assess_risk` with a real
  subprocess + tmp git worktree + `pytest -x` harness.

### 4.6 RLHF — `src/chimera/rlhf/`

- `RewardModel` (`reward_model.py`) — distilbert fine-tuned on preference pairs.
- `RLHFOracle` (`oracle.py`) — scores / ranks candidate responses.
- **Verdict:** directly reusable. Phase 1 optional (QA could use it to score
  candidate verdicts); Phase 3 uses it to grade inter-agent work products.

### 4.7 Tests — `agi-project/tests/`

- `test_agent_logic.py` has the patterns we mirrored: `FakeLanceDB`,
  `FakeSentenceTransformer`, `MockCognitiveCore`, `fake_embedding_model`
  autouse fixture.
- For `chimera.org`, we extended `MockCognitiveCore` into
  `RoleAwareMockCognitiveCore` that dispatches canned responses by the
  `[[ROLE:Name]]` marker injected into role prompts.

### 4.8 Knight Medicare integration

- `chimera-bridge/` FastAPI service pattern is mentioned in
  `agi_project_collaboration.md` but doesn't exist in this repo.
- Phase 5 establishes the pattern locally with `org/api_service.py`.

### 4.9 Config / API keys

- `CHIMERA_LLM_API_KEY` (required) — Gemini API key.
- `CHIMERA_LLM_API_URL` (optional) — defaults to Gemini v1beta.
- Phase 1 adds `CHIMERA_ORG_DB_ROOT` (defaults to `./chimera_org_db`).

### 4.10 Prior thinking on multi-agent (already in this repo)

- `agi_project_collaboration.md` mentions RLHF + experiential self-critique and
  a consciousness → self-improvement loop.
- `project_roadmap.md` Phase 3 (planned) mentions a "multi-agent inner council"
  with assessment / intervention / empathy / safety sub-agents — this aligns
  with our CEO/R&D/Marketing/Production/Ops/QA vision.

**Summary table:**

| Subsystem | Reusable? | Notes |
|-----------|-----------|-------|
| Cognitive Core | Yes | Share one Prometheus instance |
| Agent Loop | Yes (compose, don't drive) | Role wraps Agent; Org drives |
| Memory | Yes | One db_path per role |
| ToolRegistry | Yes | Per-role registries |
| Consciousness | Partial | SelfSimulationFramework needs real predictor for Phase 4 |
| RLHF Oracle | Yes | Phase 3 grader |
| Tests | Yes | Extend FakeLanceDB / MockCognitiveCore patterns |
| FastAPI bridge | Missing | Build in Phase 5 |
| Config / keys | Adequate | Add org-specific vars |

---

## 5. Synthesis — Where the Research Landed in `chimera.org`

This section ties each cherry-picked idea to a concrete file in
`agi-project/src/chimera/org/`.

| Idea source | Pattern | Where it lives |
|-------------|---------|----------------|
| MetaGPT role SOPs | Role-specific system prompts with structured output | `prompts.py` |
| CrewAI agent shape | `Role` with `name`, `allowed_tools`, `next_role`, `process(wo)` | `role.py` + `roles/*.py` |
| CrewAI process modes | Sequential mode in Phase 1; hierarchical in Phase 4 | `org.py:Org.run_until_complete` |
| LangGraph state machine | Explicit `OrgStatus` enum + transitions table | `work_order.py:_LEGAL_TRANSITIONS` |
| LangGraph checkpointing | SQLite-backed durable store; `Org.resume()` | `store.py` + `org.py:resume` |
| SWE-agent CI gate (Phase 4) | Subprocess + tmp worktree + `pytest -x` before PR | `consciousness/narcissus_core.py:SelfSimulationFramework` (deferred) |
| Voyager skill library (Phase 3) | `org.skills` namespace in `VectorEpisodicMemory` | deferred — Phase 3 |
| Reflexion (Phase 3) | Failure post-mortem as immutable memory row | deferred — Phase 3 |
| METR audit log | Append-only `WorkOrder.history` | `work_order.py:WorkOrder.history` |
| HITL approval | `ApprovalGate` (Phase 1 pass-through, Phase 2+ enforced) | `approval.py` |

**Explicitly skipped in Phase 1** (because the plan said "smallest demoable
slice first"):

- No Gmail / Calendar / Drive / GitHub MCP calls — Phase 2+.
- No skill library / Voyager curriculum — Phase 3.
- No self-modification or SelfSimulationFramework integration — Phase 4.
- No runtime role-spawning — Phase 4+.
- No FastAPI dashboard or Vercel — Phase 5.
- No hierarchical / consensual process modes — sequential only.
- No async or concurrency — one WorkOrder, one thread, blocking calls.

---

## 6. Phased Rollout (Summary)

### Phase 1 — Six-role skeleton, in-process **(shipped in this PR)**

CLI submits a goal → 6 roles touch it in order → final WorkOrder prints as
JSON. No external I/O. Uses `MockCognitiveCore` for tests,
`PrometheusCognitiveCore` for smoke runs.

### Phase 2 — External I/O + real approval gate

`InboxAdapter` (Gmail MCP poller), `CalendarAdapter`, `DriveAdapter`.
`ApprovalGate` enforced for all outbound effects. Outbound = Gmail **draft**,
never `send`. Demo: email tagged `chimera/intake` lands in your inbox → org
produces a draft reply within 60s.

### Phase 3 — Skill library + Reflexion

`org/skill_library.py` using a `VectorEpisodicMemory` namespace `org.skills`.
Each successful WorkOrder spawns a `Skill(name, description, code_or_template,
success_rate)` row. Roles recall top-k skills as few-shot context. Failed
WorkOrders produce a `Reflection` row in `org.reflections` (immutable).

### Phase 4 — Self-improvement loop + runtime role-spawning

New `Researcher` role can propose code diffs targeting `chimera/org/`.
Pipeline: propose → `SelfSimulationFramework.simulate_cognitive_change`
(real subprocess + tmp git worktree + `pytest -x` harness) → `ApprovalGate` →
`mcp__github__create_branch` + `create_or_update_file` +
`create_pull_request`. PRs target `claude/agent-company-ecosystem-*` only,
never `master`, auto-merge permanently off. Runtime role-spawning gated by
`max_concurrent_roles` (default 12) and `max_spawned_per_hour` (default 3).

### Phase 5 — FastAPI dashboard

`org/api_service.py` exposes `/workorders`, `/workorders/{id}`, `/approvals`.
HTMX frontend, SSE live tail. Deploys via Vercel MCP.

---

## 7. References

### Multi-agent frameworks
- MetaGPT — https://github.com/geekan/MetaGPT
- CrewAI — https://github.com/crewAIInc/crewAI
- LangGraph — https://github.com/langchain-ai/langgraph
- ChatDev — https://github.com/OpenBMB/ChatDev
- CAMEL — https://github.com/camel-ai/camel
- AgentVerse — https://github.com/OpenBMB/AgentVerse
- GPTeam — https://github.com/101dotxyz/GPTeam
- AutoGen — https://github.com/microsoft/autogen
- AutoGPT — https://github.com/Significant-Gravitas/AutoGPT

### Self-improving agents
- SWE-agent — https://github.com/SWE-agent/SWE-agent
- Voyager — https://github.com/minedojo/voyager
- Reflexion — https://github.com/noahshinn/reflexion
- AI-Scientist — https://github.com/sakanaai/ai-scientist
- METR Task Standard — https://github.com/METR/task-standard
- Awesome-Self-Evolving-Agents — https://github.com/XMUDeepLIT/Awesome-Self-Evolving-Agents
- OpenHands — https://github.com/All-Hands-AI/OpenHands
- Aider — https://github.com/paul-gauthier/aider

### Papers / standards
- Reflexion: Language Agents with Verbal Reinforcement Learning —
  https://openreview.net/pdf?id=vAElhFcKW6
- Voyager: An Open-Ended Embodied Agent with Large Language Models
- METR Task Standard for agent evaluation
