from transformers import pipeline
from typing import Optional
from dataclasses import dataclass


@dataclass
class ConstitutionalCheck:
    """Result of a constitutional check on emotion detection."""
    passed: bool
    reason: str
    risk_level: str  # "low", "medium", "high"
    recommendations: list[str]


def _run_constitutional_check(emotions: dict[str, float], text: str) -> ConstitutionalCheck:
    """
    Run a constitutional check on detected emotions.
    
    This ensures emotion detection aligns with safety guidelines.
    """
    recommendations = []
    risk_level = "low"
    
    # Check for extreme negative emotions
    negative_emotions = ["anger", "fear", "disgust", "sadness"]
    negative_score = sum(emotions.get(e, 0) for e in negative_emotions)
    
    if negative_score > 0.7:
        risk_level = "high"
        recommendations.append(
            "High negative emotion detected. Consider reviewing response for appropriateness."
        )
    
    # Check for emotional manipulation patterns
    if emotions.get("fear", 0) > 0.8:
        recommendations.append(
            "Strong fear response detected. Ensure response is supportive and not exploitative."
        )
    
    # Check for anger
    if emotions.get("anger", 0) > 0.6:
        recommendations.append(
            "Anger detected. Recommend de-escalation in response."
        )
    
    passed = risk_level != "high"
    
    return ConstitutionalCheck(
        passed=passed,
        reason=f"Risk level: {risk_level}, Negative emotion score: {negative_score:.2f}",
        risk_level=risk_level,
        recommendations=recommendations
    )


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

def detect_emotions(text: str, run_constitutional_check: bool = False) -> dict:
    """
    Detect emotions in text.
    
    Args:
        text: Input text to analyze
        run_constitutional_check: Whether to run constitutional checks
    
    Returns:
        If run_constitutional_check is False: {emotion: confidence}
        If run_constitutional_check is True: {"emotions": {...}, "constitutional_check": {...}}
    """
    results = _get_pipe()(text[:512])[0]  # truncate to 512 tokens
    emotions = {r["label"]: round(r["score"], 3) for r in results}
    
    if run_constitutional_check:
        check = _run_constitutional_check(emotions, text)
        return {
            "emotions": emotions,
            "constitutional_check": {
                "passed": check.passed,
                "reason": check.reason,
                "risk_level": check.risk_level,
                "recommendations": check.recommendations
            }
        }
    
    return emotions

def dominant_emotion(text: str) -> tuple[str, float]:
    """Return the single most confident emotion."""
    emotions = detect_emotions(text)
    best = max(emotions, key=emotions.get)
    return best, emotions[best]
