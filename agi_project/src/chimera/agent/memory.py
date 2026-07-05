# This file will define the agent's memory systems.

import json
from collections import deque
import os
from typing import Any, List, NamedTuple
# Placeholder for protobuf messages
# from ..protos import core_pb2


def _load_vector_dependencies():
    try:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
    except Exception as exc:
        raise ImportError(
            "VectorEpisodicMemory requires langchain_ollama and chromadb (Chroma). "
            "Install the appropriate dependencies (e.g. requirements-submodule.txt) to enable vector memory."
        ) from exc

    return Chroma, OllamaEmbeddings

class Experience(NamedTuple):
    """Represents a single experience tuple for the agent."""
    observation: Any # Should be core_pb2.Observation
    action: Any # Should be core_pb2.AgentAction
    outcome: Any # Should be core_pb2.Observation

class WorkingMemory:
    """Manages the agent's short-term context for the current task."""

    def __init__(self, max_size: int = 20):
        self.history = deque(maxlen=max_size)

    def add(self, record: Any):
        """Adds a new observation or action to the working memory."""
        self.history.append(record)

    def get_context(self) -> List[Any]:
        """Returns the current context as a list."""
        return list(self.history)

    def clear(self):
        """Clears the working memory."""
        self.history.clear()

class VectorEpisodicMemory:
    """Manages the agent's long-term, searchable memory of past experiences using LangChain Chroma."""

    def __init__(self, persist_path: str, collection_name: str = "experiences"):
        """
        Initializes the Chroma-backed episodic memory.

        Args:
            persist_path: Directory where Chroma will persist its data.
            collection_name: Name of the Chroma collection to store experiences.
        """
        print("Initializing VectorEpisodicMemory (Chroma)...")
        Chroma, SentenceTransformerEmbeddings = _load_vector_dependencies()

        # Lightweight Ollama embedding model for local use
        self.embeddings = SentenceTransformerEmbeddings(model="qwen3-embedding:0.6b")

        self.persist_path = os.path.join(persist_path, "chroma")
        os.makedirs(self.persist_path, exist_ok=True)

        # Create/open the Chroma vectorstore
        self.store = Chroma(persist_directory=self.persist_path,
                            collection_name=collection_name,
                            embedding_function=self.embeddings)

        print("VectorEpisodicMemory (Chroma) initialized successfully.")

    def _experience_to_text(self, experience: Experience) -> str:
        """Converts an Experience object into a single string for embedding."""
        obs = json.dumps(experience.observation)
        act = json.dumps(experience.action)
        out = json.dumps(experience.outcome)
        return f"Observation: {obs}\nAction: {act}\nOutcome: {out}"

    def remember(self, experience: Experience, doc_id: str = None):
        """Stores a new experience in the Chroma vectorstore.

        Args:
            experience: The Experience tuple to store.
            doc_id: Optional unique id for the document in Chroma.
        """
        text = self._experience_to_text(experience)
        metadata = {
            "observation_text": json.dumps(experience.observation),
            "action_text": json.dumps(experience.action),
            "outcome_text": json.dumps(experience.outcome)
        }

        # Chroma will compute embeddings via the provided embedding function
        if doc_id is not None:
            self.store.add_texts([text], metadatas=[metadata], ids=[doc_id])
        else:
            self.store.add_texts([text], metadatas=[metadata])

        # Persist to disk if supported
        try:
            self.store.persist()
        except Exception:
            pass

        print("--- Remembered new experience (Chroma) ---")

    def recall(self, query: str, top_k: int = 5) -> List[Experience]:
        """Recalls the most relevant experiences based on semantic similarity using Chroma."""
        # If there are no documents, return empty
        try:
            # similarity_search returns a list of langchain Document objects
            results = self.store.similarity_search(query, k=top_k)
        except Exception:
            return []

        recalled_experiences: List[Experience] = []
        for doc in results:
            md = doc.metadata or {}
            exp = Experience(
                observation=json.loads(md.get("observation_text", "null") or "null"),
                action=json.loads(md.get("action_text", "null") or "null"),
                outcome=json.loads(md.get("outcome_text", "null") or "null")
            )
            recalled_experiences.append(exp)

        print(f"--- Recalled {len(recalled_experiences)} experiences for query: '{query}' ---")
        return recalled_experiences
