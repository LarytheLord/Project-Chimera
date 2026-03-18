"""
Integration module to connect Narcissus consciousness system with Project Chimera's agent
"""
from typing import Any, Dict, List

from .narcissus_core import NarcissusConsciousnessCore, CognitiveState


class ConsciousnessIntegration:
    """Bridges the consciousness system with the main agent functionality"""
    
    def __init__(self, consciousness_core: NarcissusConsciousnessCore):
        self.consciousness = consciousness_core
        self.consciousness_enabled = True
        
    def enable_consciousness_monitoring(self):
        """Enable the consciousness monitoring system"""
        self.consciousness.is_monitoring = True
        
    def disable_consciousness_monitoring(self):
        """Disable the consciousness monitoring system (for performance)"""
        self.consciousness.is_monitoring = False
        
    def record_cognitive_state_from_agent(self, 
                                        thought_process: str,
                                        attention_weights: Dict[str, float], 
                                        decision_path: List[str],
                                        confidence: float,
                                        memory_context: List[str],
                                        processing_load: float,
                                        emotional_state: Dict[str, float] = None):
        """Record cognitive state from the agent's perspective"""
        if not self.consciousness_enabled:
            return None
            
        if emotional_state is None:
            emotional_state = {"curiosity": 0.7, "confidence": confidence, "focus": 0.8}
        
        return self.consciousness.record_cognitive_state(
            thought_process=thought_process,
            attention_weights=attention_weights,
            decision_path=decision_path,
            confidence=confidence,
            emotional_state=emotional_state,
            memory_context=memory_context,
            processing_load=processing_load
        )
    
    def get_consciousness_insights(self) -> Dict[str, Any]:
        """Get insights from the consciousness system for agent decision making"""
        if not self.consciousness_enabled:
            return {}
            
        return self.consciousness.perform_introspective_analysis()
    
    def simulate_cognitive_change(self, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Use consciousness system to simulate a cognitive change"""
        if not self.consciousness_enabled:
            return {}
            
        return self.consciousness.simulate_self_modification(description, parameters)
