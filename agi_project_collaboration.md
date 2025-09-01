# Project Chimera: Collaboration Guide

This document provides an overview of the Project Chimera's current state, recent enhancements, and instructions for setting up the development environment and running tests.

## Project Overview

Project Chimera is an AGI system designed with a modular architecture, separating cognitive core, agent logic, and tool usage. The goal is to create a robust and extensible AGI framework.

## Recent Enhancements

### Advanced Tooling System

We have moved beyond simple tool definitions to a more robust, schema-driven system. This is a critical step towards our goal of an agent that can learn to use new tools autonomously.

**Key Changes:**
- **Tool Schemas:** Every tool now exposes a `get_schema()` method. This method returns a JSON schema that explicitly defines the tool's arguments, their types, and whether they are required. This structured information allows the agent to reliably construct valid tool calls.
- **`FileSystemTool`:** We have introduced a new `FileSystemTool` that empowers the agent to interact with its local environment. This tool currently supports `list_directory` and `read_file` operations, laying the groundwork for the agent to eventually read, write, and even modify its own source code.
- **Agent Prompting:** The agent's core prompt has been updated to include the new tool schemas. It is now explicitly instructed to use these schemas to format its tool calls as JSON objects.

### Prometheus Engine: The Gift of Fire

We have replaced the `MockCognitiveCore` with the **Prometheus Engine**, a new cognitive core that connects our agent to a real, powerful language model. This is a major step towards true intelligence, allowing the agent to move beyond scripted responses to generative, nuanced thought.

**Key Changes:**
- **`PrometheusCognitiveCore`:** A new class in `src/cognitive_core/prometheus_core.py` that handles communication with external language model APIs.
- **API Key Management:** The engine requires an API key, which must be provided via the `GEMINI_API_KEY` environment variable.
- **`httpx` Dependency:** We have added the `httpx` library to manage API calls.

### Memory Persistence

We have recently enhanced the agent's memory system to include persistence and recall. The `Agent` now utilizes `EpisodicMemory` to store and retrieve past experiences from a file, allowing the agent to maintain context across sessions.

**Key Changes:**
- The `Agent` constructor now accepts an optional `memory_filepath` parameter.
- `EpisodicMemory` handles reading from and writing to this file.
- The agent's `run_main_loop` now includes a "Remember" step to store experiences.

## Setting Up the Development Environment

To get started with Project Chimera, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd agi
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd agi-project
    ```

3.  **Install dependencies:**
    We use `poetry` for dependency management. If you don't have `poetry` installed, follow the instructions [here](https://python-poetry.org/docs/#installation).
    ```bash
    poetry install
    ```

4.  **Activate the virtual environment:**
    ```bash
    poetry shell
    ```

## Running Tests and Verifying Changes

To test the agent's capabilities, especially the new `FileSystemTool`, you can use the `MockCognitiveCore`.

1.  **Modify `src/main.py`:** Temporarily replace the `PrometheusCognitiveCore` with the `MockCognitiveCore`. The `MockCognitiveCore` is hardcoded to return a specific tool call, allowing you to test tool functionality without needing a live API key.
2.  **Run the agent:**
    ```bash
    python src/main.py
    ```
    The agent will execute the hardcoded tool call and you can observe the output.

## Contributing

-   **Adhere to existing conventions:** When making changes, please follow the established coding style, structure, and architectural patterns.
-   **Write tests:** For new features or bug fixes, ensure adequate test coverage.
-   **Document changes:** Update relevant documentation for any significant changes.

Let's build something extraordinary together!

### RLHF with Experiential Self-Critique: Towards Autonomous Learning

We have implemented an innovative Reinforcement Learning from Human Feedback (RLHF) system designed to enable the agent to learn and improve its behavior with minimal direct human supervision. This approach, termed "Experiential Self-Critique," allows the agent to leverage its own experiences to refine its actions.

**Key Components & Workflow:**

1.  **Preference Data Collection (`scripts/collect_preferences.py`):
    *   An interactive script allows human users to provide prompts and choose between two agent-generated responses (one "chosen," one "rejected"). This initial human feedback seeds the system with basic preferences.

2.  **Reward Model (`src/rlhf/reward_model.py`):
    *   A lightweight `distilbert-base-uncased` model is fine-tuned on the collected preference data. This model learns to assign a scalar "reward score" to any given (prompt, response) pair, reflecting human preferences.
    *   A dedicated script (`scripts/train_reward_model.py`) orchestrates the training of this model.

3.  **RLHF Oracle (`src/rlhf/oracle.py`):
    *   This component loads the trained Reward Model and acts as an "oracle" for the agent. It can score individual responses and, crucially, select the best response from a list of candidates based on the learned preferences.

4.  **Agent Integration (`src/agent/agent.py`):
    *   The agent's `_think` process has been enhanced:
        *   For a given observation, the agent now generates multiple candidate actions/responses.
        *   It then consults the `RLHFOracle` to evaluate these candidates and select the one with the highest reward score.
        *   *(Future Work: Automated Preference Generation)* This selected "best" response can then be automatically paired with a "rejected" (lower-scoring) candidate to generate new preference data, which can be fed back into the Reward Model training, creating a continuous, self-improving learning loop.

**Why this Stands Out (The "Chimera" Approach):**

*   **Reduced Human Bottleneck:** By enabling the agent to critique its own generated responses using the Reward Model, we significantly reduce the need for extensive manual human labeling, making the learning process more scalable.
*   **Leverages Existing Memory:** The system intelligently integrates with the `VectorEpisodicMemory`, allowing the agent to draw upon past experiences during its self-critique process.
*   **Resource-Efficient:** All components are designed to run efficiently on CPU-only hardware, adhering to our project's resource constraints.
*   **Path to Autonomous Learning:** This architecture lays the groundwork for the agent to become more introspective and autonomously improve its decision-making and behavior over time, aligning with the long-term vision of Project Chimera.