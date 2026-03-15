import json
from pathlib import Path
from typing import Dict, List

from chimera.rlhf.reward_model import RewardModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_preference_data(filepath: str) -> List[Dict[str, str]]:
    """Loads preference data from a JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    """Main function to run the reward model training script."""
    print("--- Starting Reward Model Training Script ---")

    # 1. Define file paths
    preference_file = PROJECT_ROOT / "preference_data.jsonl"
    output_dir = PROJECT_ROOT / "trained_models" / "reward_model"

    if not preference_file.exists():
        print(f"Error: Preference data file not found at {preference_file}")
        print("Please run scripts/collect_preferences.py first to generate data.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load the data
    print(f"Loading preference data from {preference_file}...")
    preference_data = load_preference_data(str(preference_file))
    if not preference_data:
        print("Error: No data found in preference file. Nothing to train on.")
        return

    # 3. Initialize and train the model
    # Using a small, efficient model for our resource constraints
    reward_model = RewardModel(model_name="distilbert-base-uncased")
    
    # For demonstration, we'll run for a small number of epochs
    reward_model.train(preference_data, output_dir=str(output_dir), epochs=1)

    # 4. Save the final model
    # The trainer already saves checkpoints, but this saves the final state explicitly.
    reward_model.save(str(output_dir))

    print("--- Reward Model Training Finished Successfully ---")
    print(f"Trained model is saved in: {output_dir}")

if __name__ == "__main__":
    main()
