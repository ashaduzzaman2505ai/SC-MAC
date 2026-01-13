# src/agents/worker.py
import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LogicWorker:
    def __init__(self, model_id: str, device: str = "cuda"):
        # 1. Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 2. Load with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto", # Let accelerate handle the mapping
            low_cpu_mem_usage=True
        )
        self.device = device

    def generate_step(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Generates a single 'thought' step."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Only keep the last 1024 tokens to save memory
        if inputs['input_ids'].shape[1] > 1024:
            inputs['input_ids'] = inputs['input_ids'][:, -1024:]
            inputs['attention_mask'] = inputs['attention_mask'][:, -1024:]
                    
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
