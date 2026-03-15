# Project Chimera: An Artificial General Intelligence

This repository contains the source code and documentation for Project Chimera, a phased initiative to develop a self-sufficient Artificial General Intelligence (AGI).
By Abi and his team

## Package Usage

Chimera is structured as a reusable Python package under `src/chimera`.

```python
from chimera.agent.memory import VectorEpisodicMemory
from chimera.cognitive_core.prometheus_core import PrometheusCognitiveCore
```

Set `CHIMERA_LLM_API_KEY` before using the remote Prometheus cognitive core.
## Plan

The project follows a multi-phase development plan:

*   **Phase 0: Foundation & Scaffolding:** Set up the project structure, environment, and core interfaces.
*   **Phase 1: The Cognitive Core ("Prometheus"):** Build the core multi-modal, mixture-of-experts neural model.
*   **Phase 2: The Agentic Framework ("Janus"):** Give the core model the ability to use tools, act, and learn from feedback.
*   **Phase 3: Self-Improvement & Metacognition ("Ouroboros"):** Enable the AGI to analyze and rewrite its own code to improve its capabilities.

## Tech Stack

*   **Core Language:** Python
*   **Performance Modules:** Rust
*   **ML Framework:** JAX
*   **Data Interchange:** Apache Arrow
*   **Deployment:** Docker & Kubernetes
