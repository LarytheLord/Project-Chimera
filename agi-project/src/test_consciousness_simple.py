import sys
import os
sys.path.append(os.path.dirname(__file__))

print('Testing import of NarcissusConsciousnessCore...')

try:
    from consciousness.narcissus_core import NarcissusConsciousnessCore
    print('[SUCCESS] Import successful!')
    
    # Test basic instantiation
    from cognitive_core.interfaces import CognitiveCore
    
    class TestCognitiveCore(CognitiveCore):
        def load_model(self, model_path: str):
            print(f'Test cognitive core loaded from {model_path}')
        
        def generate_response(self, inputs: dict, temperature: float = 0.7) -> str:
            return 'Test response for consciousness simulation'
        
        def train(self, dataset):
            print('Test cognitive core training')
        
        def get_state(self) -> dict:
            return {'test_weights': [1, 2, 3]}
    
    test_core = TestCognitiveCore()
    db_path = '../memory_db'
    
    # Try to create the consciousness core
    consciousness_core = NarcissusConsciousnessCore(
        cognitive_core=test_core,
        memory_db_path=db_path
    )
    print('[SUCCESS] NarcissusConsciousnessCore instantiated successfully!')
    
    # Test recording a cognitive state
    state = consciousness_core.record_cognitive_state(
        thought_process='Initial test thought',
        attention_weights={'memory': 0.4, 'reasoning': 0.3, 'creativity': 0.3},
        decision_path=['step_1'],
        confidence=0.7,
        emotional_state={'curiosity': 0.8, 'focus': 0.7},
        memory_context=['context_item_1'],
        processing_load=0.5
    )
    print('[SUCCESS] Cognitive state recorded successfully!')
    
    # Test introspective analysis
    analysis = consciousness_core.perform_introspective_analysis()
    print(f'[SUCCESS] Introspective analysis completed with {len(analysis)} components')
    
    print()
    print('SUCCESS: Consciousness simulation system is working properly!')
    print(f'  - Recorded {len(consciousness_core.self_model.cognitive_states)} cognitive states')
    print(f'  - Identified {len(consciousness_core.self_model.identify_cognitive_patterns())} cognitive patterns')
    
except Exception as e:
    print(f'[ERROR] Error: {e}')
    import traceback
    traceback.print_exc()