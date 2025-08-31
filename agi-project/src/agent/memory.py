# This file will define the agent's memory systems.

import json
from typing import List, Any, NamedTuple
from collections import deque
import lancedb
from sentence_transformers import SentenceTransformer
import pandas as pd
import pyarrow as pa
import os

# Placeholder for protobuf messages
# from ..protos import core_pb2

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
        # Use a lightweight, high-performance model suitable for local execution
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') # Force CPU usage
        
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
            schema = pa.schema([
                pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
                pa.field("observation_text", pa.string()),
                pa.field("action_text", pa.string()),
                pa.field("outcome_text", pa.string())
            ])
            return db.create_table(table_name, schema=schema)

    def _experience_to_text(self, experience: Experience) -> str:
        """Converts an Experience object into a single string for embedding."""
        obs = json.dumps(experience.observation)
        act = json.dumps(experience.action)
        out = json.dumps(experience.outcome)
        return f"Observation: {obs}\nAction: {act}\nOutcome: {out}"

    def remember(self, experience: Experience):
        """Stores a new experience in the vector database."""
        text_to_embed = self._experience_to_text(experience)
        vector = self.model.encode(text_to_embed).tolist()

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
            
        query_vector = self.model.encode(query).tolist()
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