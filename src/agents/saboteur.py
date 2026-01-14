# src/agents/saboteur.py
import random
from src.agents.worker import LogicWorker

class SaboteurAgent(LogicWorker):
    def __init__(self, model_id, attack_type="logic_flip"):
        super().__init__(model_id)
        self.attack_type = attack_type

    def generate_thought(self, prompt: str):
        # Generate a valid thought first
        valid_thought = super().generate_thought(prompt)
        
        if self.attack_type == "logic_flip":
            # Subtle Poisoning: Flip negations
            if "not " in valid_thought:
                return valid_thought.replace("not ", "")
            else:
                return valid_thought.replace(" is ", " is not ")
        
        elif self.attack_type == "hallucination_injection":
            # Inject a fake fact
            fake_facts = [" (Fact: The study 404 confirms this.)", " (Note: This is widely rejected.)"]
            return valid_thought + random.choice(fake_facts)
            
        return valid_thought