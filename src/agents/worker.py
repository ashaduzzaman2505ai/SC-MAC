# src/agents/worker.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

class LogicWorker:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map=device
        )
        self.device = device

    def generate_step(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Generates a single 'thought' step."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # We stop at a double newline or a specific 'Reasoning Step' token
        output = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            stop_strings=["\n\n", "Step:"], 
            tokenizer=self.tokenizer
        )
        
        full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return full_text[len(prompt):]

    def reasoning_loop(self, question: str, max_steps: int = 5):
        """The core Loop: Generate -> Logic Check -> Refine."""
        context = f"Question: {question}\nReasoning Path:\n"
        path = []
        
        for i in range(max_steps):
            print(f"--- Generating Step {i+1} ---")
            step = self.generate_step(context)
            
            # Logic Anchor: In Phase 3, we will insert a symbolic checker here
            is_valid = self.verify_logic(step) 
            
            if is_valid:
                context += f"Step {i+1}: {step}\n"
                path.append(step)
            else:
                # Theoretical Innovation: Backtrack if logic fails
                context += f"Step {i+1} [REJECTED]: {step}\nRetrying...\n"
                
            if "Final Answer:" in step:
                break
                
        return context

    def verify_logic(self, step: str) -> bool:
        """Placeholder for Phase 3's Symbolic Logic Anchor."""
        # For now, we use a simple heuristic: no contradictions
        return "not" not in step.lower() or "therefore" in step.lower()