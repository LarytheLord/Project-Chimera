# This file will define the agent's memory systems.

import json
from typing import List, Tuple, Any, NamedTuple
from collections import deque

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

class EpisodicMemory:
    """Manages the agent's long-term, searchable memory of past experiences.
    
    In a real implementation, this would be backed by a vector database like ChromaDB or Pinecone
    to allow for efficient similarity-based retrieval of relevant memories.
    """

    def __init__(self, filepath: str = None):
        self.experiences: List[Experience] = []
        self.filepath = filepath
        if self.filepath:
            self.load_from_file(self.filepath)

    def remember(self, experience: Experience):
        """Stores a new experience in long-term memory and saves it if a file is configured."""
        self.experiences.append(experience)
        if self.filepath:
            self.save_to_file(self.filepath, experience)

    def recall(self, query: str, top_k: int = 5) -> List[Experience]:
        """Recalls the most relevant experiences based on a query.
        
        This is a simplified keyword-based search. A real implementation would use
        vector embeddings of the query and experiences to find semantic similarity.
        """
        # Naive search for demonstration purposes
        relevant_experiences = []
        for exp in self.experiences:
            # Pretend we are searching inside the observation text
            if hasattr(exp.observation, 'data') and hasattr(exp.observation.data, 'text_data'):
                if query in exp.observation.data.text_data:
                    relevant_experiences.append(exp)
        
        return relevant_experiences[-top_k:] # Return the most recent k matches

    def save_to_file(self, filepath: str, experience: Experience):
        """Appends a single experience to the JSONL file."""
        with open(filepath, 'a') as f:
            f.write(json.dumps(experience._asdict()) + '\n')

    def load_from_file(self, filepath: str):
        """Loads experiences from a JSONL file."""
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.experiences.append(Experience(**data))
        except FileNotFoundError:
            # It's okay if the file doesn't exist yet.
            pass

