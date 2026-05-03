"""
Integration module to connect Narcissus consciousness system with Project Chimera's agent
"""
from typing import Any, Dict, List, Optional

from ..eventbus import EventBus, NullEventBus, Topics
from .narcissus_core import NarcissusConsciousnessCore, CognitiveState


class ConsciousnessIntegration:
    """Bridges the consciousness system with the main agent functionality"""

    def __init__(
        self,
        consciousness_core: NarcissusConsciousnessCore,
        event_bus: Optional[EventBus] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.consciousness = consciousness_core
        self.consciousness_enabled = True
        self.event_bus: EventBus = event_bus or NullEventBus()
        self.agent_id = agent_id
        self.session_id = session_id
        
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
                                        emotional_state: Dict[str, float] = None,
                                        trace_id: Optional[str] = None):
        """Record cognitive state from the agent's perspective"""
        if not self.consciousness_enabled:
            return None

        if emotional_state is None:
            emotional_state = {"curiosity": 0.7, "confidence": confidence, "focus": 0.8}

        state = self.consciousness.record_cognitive_state(
            thought_process=thought_process,
            attention_weights=attention_weights,
            decision_path=decision_path,
            confidence=confidence,
            emotional_state=emotional_state,
            memory_context=memory_context,
            processing_load=processing_load
        )

        try:
            self.event_bus.publish(
                Topics.METACOG_REFLECTIONS,
                {
                    "thought_process": thought_process,
                    "attention_weights": attention_weights,
                    "decision_path": decision_path,
                    "confidence": confidence,
                    "emotional_state": emotional_state,
                    "processing_load": processing_load,
                    "self_reflection": getattr(state, "self_reflection", None),
                },
                trace_id=trace_id,
                session_id=self.session_id,
                agent_id=self.agent_id,
                event_type="cognitive_state",
            )
        except Exception:  # pragma: no cover - defensive
            pass

        return state
    
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
