import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.logic.verifier import SymbolicVerifier

class LogicWorker:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        print(f"🚀 Initializing SC-MAC Worker: {model_id}")
        
        # 1. Memory Optimization (4-bit NF4)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        self.verifier = SymbolicVerifier()

    def flush_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def generate_thought(self, prompt: str):
        """Generates a single reasoning step with a stop criteria."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # 2026 HF update: Using 'stop_strings' to end generation at 'Step:'
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            stop_strings=["\n", "Step:"], 
            tokenizer=self.tokenizer
        )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text[len(prompt):].strip()

    def run_sc_mac_loop(self, question: str):
        self.flush_memory()
        context = f"Question: {question}\nReasoning Path:\n"
        path = []

        for i in range(1, 6): # Max 5 reasoning steps
            print(f"🔍 Reasoning Step {i}...")
            step = self.generate_thought(context)
            
            # Logic Anchor Check
            if self.verifier.check_consistency(step):
                print(f"✅ Step {i} Accepted.")
                context += f"Step {i}: {step}\n"
                path.append(step)
                # If the agent declares a final answer, stop
                if "Final Answer" in step: break
            else:
                print(f"❌ Step {i} Rejected (Logic Violation). Retrying...")
                context += f"Step {i} [Correction]: Let me re-evaluate...\n"
            
            self.flush_memory()

        return context

    def get_thought_embedding(self, step_text: str):
        """Extracts the latent vector for a specific reasoning step."""
        inputs = self.tokenizer(step_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get the last hidden layer for the last token
            # Note: For 4-bit models, this remains in float16/bfloat16
            embedding = outputs.hidden_states[-1][:, -1, :]
        return F.normalize(embedding, p=2, dim=-1)