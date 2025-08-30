# Project Chimera: Collaboration Guide

This document provides an overview of the Project Chimera's current state, recent enhancements, and instructions for setting up the development environment and running tests.

## Project Overview

Project Chimera is an AGI system designed with a modular architecture, separating cognitive core, agent logic, and tool usage. The goal is to create a robust and extensible AGI framework.

## Recent Enhancements

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

## Running Tests

We have updated the test harness to accommodate the new memory persistence functionality.

**To run the tests:**

1.  **Ensure you are in the `agi-project` directory and the poetry shell is activated.**
2.  **Execute the test command:**
    ```bash
    python -m pytest tests/test_agent_logic.py
    ```
    *Note: If you encounter import errors, ensure your `PYTHONPATH` includes the `src` directory within `agi-project` or run `pip install -e .` from the `agi-project` directory to install it in editable mode.*

## Contributing

-   **Adhere to existing conventions:** When making changes, please follow the established coding style, structure, and architectural patterns.
-   **Write tests:** For new features or bug fixes, ensure adequate test coverage.
-   **Document changes:** Update relevant documentation for any significant changes.

Let's build something extraordinary together!
