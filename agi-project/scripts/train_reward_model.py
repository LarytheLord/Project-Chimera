
import os
import sys
import json
from typing import List, Dict

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.rlhf.reward_model import RewardModel

def load_preference_data(filepath: str) -> List[Dict[str, str]]:
    """Loads preference data from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    """Main function to run the reward model training script."""
    print("--- Starting Reward Model Training Script ---")

    # 1. Define file paths
    preference_file = os.path.join(project_root, "preference_data.jsonl")
    output_dir = os.path.join(project_root, "trained_models", "reward_model")

    if not os.path.exists(preference_file):
        print(f"Error: Preference data file not found at {preference_file}")
        print("Please run scripts/collect_preferences.py first to generate data.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 2. Load the data
    print(f"Loading preference data from {preference_file}...")
    preference_data = load_preference_data(preference_file)
    if not preference_data:
        print("Error: No data found in preference file. Nothing to train on.")
        return

    # 3. Initialize and train the model
    # Using a small, efficient model for our resource constraints
    reward_model = RewardModel(model_name="distilbert-base-uncased")
    
    # For demonstration, we'll run for a small number of epochs
    reward_model.train(preference_data, output_dir=output_dir, epochs=1)

    # 4. Save the final model
    # The trainer already saves checkpoints, but this saves the final state explicitly.
    reward_model.save(output_dir)

    print("--- Reward Model Training Finished Successfully ---")
    print(f"Trained model is saved in: {output_dir}")

if __name__ == "__main__":
    main()
