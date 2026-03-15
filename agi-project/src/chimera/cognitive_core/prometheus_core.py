import httpx
import json
import os
from typing import Any, Dict

from .interfaces import CognitiveCore

DEFAULT_GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-pro:generateContent"
)


class PrometheusCognitiveCore(CognitiveCore):
    """
    A concrete implementation of the CognitiveCore that uses a real language model
    via an API call. This is the "Prometheus Engine" of our AGI.
    """

    def __init__(self, api_url: str = None, api_key: str = None):
        self.api_url = api_url or os.environ.get("CHIMERA_LLM_API_URL") or DEFAULT_GEMINI_API_URL
        self.api_key = api_key or os.environ.get("CHIMERA_LLM_API_KEY")
        if not self.api_key:
            raise ValueError("CHIMERA_LLM_API_KEY environment variable required.")
        # The API key for Gemini is not a Bearer token, it's passed as a query parameter.
        self.client = httpx.Client(params={"key": self.api_key})

    def load_model(self, model_path: str):
        """For this core, loading a model is conceptual, as the model is remote."""
        print(f"Prometheus Engine connected to remote model at {self.api_url}")

    def generate_response(self, inputs: Dict[str, Any], temperature: float = 0.7) -> str:
        """
        Generates a response from the remote language model.

        Args:
            inputs: A dictionary containing the prompt text.
            temperature: The sampling temperature for generation. Higher values (e.g., 1.0)
                         make the output more random, lower values (e.g., 0.2) make it more deterministic.
        """
        prompt = inputs.get("text_data", "")
        if not prompt:
            return "Error: No prompt provided."

        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temperature
            }
        }

        try:
            response = self.client.post(self.api_url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            response_data = response.json()
            
            # Defensive parsing of the response JSON
            candidates = response_data.get('candidates', [])
            if not candidates:
                return "Error: No candidates found in API response."
            
            content = candidates[0].get('content', {})
            parts = content.get('parts', [])
            if not parts:
                return "Error: No parts found in API response content."

            generated_text = parts[0].get('text', 'Error: No text found in API response part.')
            
            print(f"\n--- Prometheus Engine (Temp: {temperature}) generated response ---")
            print(generated_text)
            print("--- End of response ---\n")
            return generated_text

        except httpx.RequestError as e:
            return f"Error: API request failed: {e}"
        except json.JSONDecodeError:
            return f"Error: Failed to decode JSON response: {response.text}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    def train(self, dataset: Any):
        """Training is not handled via this interface for a remote model."""
        print("Training is handled offline, not through the Prometheus Engine.")

    def get_state(self) -> Any:
        """Returns the state of the connection, not model weights."""
        return {"api_url": self.api_url, "connected": True}
