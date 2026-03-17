# This file will define the agent's memory systems.

import json
from collections import deque
import os
from typing import Any, List, NamedTuple

# Placeholder for protobuf messages
# from ..protos import core_pb2


def _load_vector_dependencies():
    try:
        import lancedb
        import pyarrow as pa
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "VectorEpisodicMemory requires lancedb, pyarrow, and sentence-transformers. "
            "Install requirements-submodule.txt to enable vector memory."
        ) from exc

    return lancedb, pa, SentenceTransformer

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
    """Manages the agent's long-term, searchable memory of past experiences using a vector database."""

    def __init__(self, db_path: str, table_name: str = "experiences"):
        """
        Initializes the vector-based episodic memory.

        Args:
            db_path: Path to the LanceDB database directory.
            table_name: Name of the table to store experiences.
        """
        print("Initializing VectorEpisodicMemory...")
        lancedb, pa, sentence_transformer = _load_vector_dependencies()
        self._pa = pa
        # Use a lightweight, high-performance model suitable for local execution
        self.model = sentence_transformer('all-MiniLM-L6-v2', device='cpu') # Force CPU usage
        
        db_uri = os.path.join(db_path, "lancedb")
        os.makedirs(db_uri, exist_ok=True)
        db = lancedb.connect(db_uri)
        
        self.table = self._get_or_create_table(db, table_name)
        print("VectorEpisodicMemory initialized successfully.")

    def _get_or_create_table(self, db, table_name):
        if table_name in db.table_names():
            return db.open_table(table_name)
        else:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            schema = self._pa.schema([
                self._pa.field("vector", self._pa.list_(self._pa.float32(), embedding_dim)),
                self._pa.field("observation_text", self._pa.string()),
                self._pa.field("action_text", self._pa.string()),
                self._pa.field("outcome_text", self._pa.string())
            ])
            return db.create_table(table_name, schema=schema)

    def _experience_to_text(self, experience: Experience) -> str:
        """Converts an Experience object into a single string for embedding."""
        obs = json.dumps(experience.observation)
        act = json.dumps(experience.action)
        out = json.dumps(experience.outcome)
        return f"Observation: {obs}\nAction: {act}\nOutcome: {out}"

    def _to_vector_list(self, embedding: Any) -> List[float]:
        """Normalizes encoder outputs to a plain Python list."""
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)

    def remember(self, experience: Experience):
        """Stores a new experience in the vector database."""
        text_to_embed = self._experience_to_text(experience)
        vector = self._to_vector_list(self.model.encode(text_to_embed))

        data = {
            "vector": vector,
            "observation_text": json.dumps(experience.observation),
            "action_text": json.dumps(experience.action),
            "outcome_text": json.dumps(experience.outcome)
        }
        self.table.add([data])
        print(f"--- Remembered new experience ---")

    def recall(self, query: str, top_k: int = 5) -> List[Experience]:
        """Recalls the most relevant experiences based on semantic similarity."""
        if self.table.count_rows() == 0:
            return []
            
        query_vector = self._to_vector_list(self.model.encode(query))
        results = self.table.search(query_vector).limit(top_k).to_df()

        recalled_experiences = []
        for _, row in results.iterrows():
            exp = Experience(
                observation=json.loads(row['observation_text']),
                action=json.loads(row['action_text']),
                outcome=json.loads(row['outcome_text'])
            )
            recalled_experiences.append(exp)
        
        print(f"--- Recalled {len(recalled_experiences)} experiences for query: '{query}' ---")
        return recalled_experiences
