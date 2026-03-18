import os
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
from datasets import Dataset

class RewardModel:
    """
    A model that learns to predict a scalar reward score for a given (prompt, response) pair.
    This model is trained on human (or machine-generated) preference data using the TRL library.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = None):
        """
        Initializes the RewardModel.

        Args:
            model_name (str): The name of the pretrained model to use from Hugging Face.
            device (str, optional): The device to run the model on ('cuda' or 'cpu'). 
                                    Defaults to 'cuda' if available, else 'cpu'.
        """
        self.device = "cpu" # Force CPU usage due to hardware constraints
        
        print(f"Initializing RewardModel on device: {self.device}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # A pad token is required for reward modeling
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def get_score(self, prompt: str, response: str) -> float:
        """
        Calculates the reward score for a single prompt-response pair.

        Returns:
            float: A scalar reward score.
        """
        inputs = self.tokenizer(prompt, response, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits[0].item()

    def train(self, preference_data: List[Dict[str, str]], output_dir: str, epochs: int = 1, batch_size: int = 2, learning_rate: float = 2e-5):
        """
        Trains the reward model on a dataset of preference pairs using TRL's RewardTrainer.

        Args:
            preference_data: A list of dictionaries, each with "prompt", "chosen", and "rejected" keys.
            output_dir: The directory to save the trained model and training artifacts.
            epochs: The number of training epochs.
            batch_size: The training batch size.
            learning_rate: The learning rate for the optimizer.
        """
        print(f"Starting reward model training with {len(preference_data)} preference pairs.")

        # TRL's RewardTrainer expects a dataset with columns for chosen and rejected responses.
        # We first need to tokenize the prompts and responses.
        def tokenize_function(examples):
            prompt_plus_chosen = [p + " " + c for p, c in zip(examples["prompt"], examples["chosen"])]
            prompt_plus_rejected = [p + " " + r for p, r in zip(examples["prompt"], examples["rejected"])]
            
            tokens_chosen = self.tokenizer(prompt_plus_chosen, truncation=True, padding="max_length", max_length=512)
            tokens_rejected = self.tokenizer(prompt_plus_rejected, truncation=True, padding="max_length", max_length=512)
            
            return {
                "input_ids_chosen": tokens_chosen["input_ids"],
                "attention_mask_chosen": tokens_chosen["attention_mask"],
                "input_ids_rejected": tokens_rejected["input_ids"],
                "attention_mask_rejected": tokens_rejected["attention_mask"],
            }

        dataset = Dataset.from_list(preference_data)
        dataset = dataset.map(tokenize_function, batched=True)

        # Use RewardConfig for TRL-specific training arguments
        training_args = RewardConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=1,
            save_total_limit=2,
            remove_unused_columns=False, # Keep original columns
        )

        trainer = RewardTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        print(f"Reward model training complete. Model saved to {output_dir}")

    def save(self, model_path: str):
        """Saves the model and tokenizer to the specified path."""
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        print(f"RewardModel saved to {model_path}")

    @classmethod
    def load(cls, model_path: str, device: str = None):
        """Loads a trained RewardModel from a path."""
        reward_model = cls(model_name=model_path, device=device)
        return reward_model