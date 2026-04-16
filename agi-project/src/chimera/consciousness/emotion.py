from typing import Dict
from transformers import pipeline

class EmotionDetector:
    """Detects emotions in text using a local HuggingFace model."""
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        print(f"--- Loading Emotion Detection Model: {model_name} (CPU) ---")
        self.classifier = pipeline(
            "text-classification", 
            model=model_name, 
            return_all_scores=True,
            device=-1 # Ensure CPU
        )
        
    def detect_emotion(self, text: str) -> Dict[str, float]:
        """Detects emotions in the given text and returns a dictionary of scores."""
        try:
            # Classification returns a list of lists of dicts
            results = self.classifier(text[:512]) # Truncate to model's max length
            # Convert to a flat dictionary: {label: score}
            emotions = {res['label']: res['score'] for res in results[0]}
            return emotions
        except Exception as e:
            print(f"Error during emotion detection: {e}")
            return {"neutral": 1.0}
