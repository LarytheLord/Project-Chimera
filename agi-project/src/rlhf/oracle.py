
import os
from typing import List

from .reward_model import RewardModel

class RLHFOracle:
    """
    An oracle that uses a trained RewardModel to score and rank agent responses.
    This provides the "wisdom" for the agent's self-critique and decision-making loops.
    """

    def __init__(self, model_path: str):
        """
        Initializes the RLHFOracle.

        Args:
            model_path (str): The path to the directory containing the trained RewardModel.
        """
        print(f"Loading RLHFOracle from trained model at: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained reward model not found at {model_path}. Please run scripts/train_reward_model.py first.")
        
        self.reward_model = RewardModel.load(model_path)
        print("RLHFOracle loaded successfully.")

    def score_response(self, prompt: str, response: str) -> float:
        """
        Scores a single response for a given prompt.

        Returns:
            float: The reward score.
        """
        return self.reward_model.get_score(prompt, response)

    def get_best_response(self, prompt: str, responses: List[str]) -> str:
        """
        Finds the best response from a list of candidates based on the reward model's score.

        Returns:
            str: The response with the highest score.
        """
        if not responses:
            return None

        scored_responses = [(self.score_response(prompt, r), r) for r in responses]
        scored_responses.sort(key=lambda x: x[0], reverse=True) # Sort by score descending
        
        return scored_responses[0][1] # Return the text of the best response
