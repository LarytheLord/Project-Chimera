# This file will define the abstract interfaces for interacting with the cognitive core.

import abc
from typing import Any, Dict

# Placeholder for the actual data input type after protobuf compilation
# from ..protos import core_pb2

class CognitiveCore(abc.ABC):
    """Abstract Base Class for the Cognitive Core.
    
    This interface defines the contract for how the agent interacts with the underlying
    generative model (e.g., PrometheusModel).
    """

    @abc.abstractmethod
    def load_model(self, model_path: str):
        """Loads the model weights from a specified path."""
        pass

    @abc.abstractmethod
    def generate_response(self, inputs: Dict[str, Any]) -> str:
        """Generates a response from the model based on the given inputs."""
        pass

    @abc.abstractmethod
    def train(self, dataset: Any):
        """Trains the model on a given dataset."""
        pass

    @abc.abstractmethod
    def get_state(self) -> Any:
        """Returns the current state of the model (e.g., weights)."""
        pass
