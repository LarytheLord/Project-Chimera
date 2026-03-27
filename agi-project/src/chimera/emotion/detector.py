from transformers import pipeline
from typing import Optional

_pipe = None

def _get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=-1,  # CPU
        )
    return _pipe

def detect_emotions(text: str) -> dict[str, float]:
    """Detect emotions in text. Returns {emotion: confidence}."""
    results = _get_pipe()(text[:512])[0]  # truncate to 512 tokens
    return {r["label"]: round(r["score"], 3) for r in results}

def dominant_emotion(text: str) -> tuple[str, float]:
    """Return the single most confident emotion."""
    emotions = detect_emotions(text)
    best = max(emotions, key=emotions.get)
    return best, emotions[best]
