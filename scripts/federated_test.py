import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.worker import LogicWorker
from src.federated.aggregator import FederatedConsensus

def run_federated_consensus_test():
    # 1. Setup
    shared_model = LogicWorker() # Only one model in VRAM
    consensus_engine = FederatedConsensus(threshold=0.88)
    
    query = "Is the vaccine safe for patients with Condition X based on local clinic data?"
    
    # 2. Simulate Agent A (Knowledgeable)
    print("🤖 Agent A Reasoning...")
    step_a = "Based on clinical trial 402, Condition X shows no adverse reactions."
    emb_a = shared_model.get_thought_embedding(step_a)
    
    # 3. Simulate Agent B (Hallucinating or Differing)
    print("🤖 Agent B Reasoning...")
    step_b = "Most patients are fine, but some reports suggest a conflict with Condition X."
    emb_b = shared_model.get_thought_embedding(step_b)
    
    # 4. Compute Latent Consensus
    score = consensus_engine.compute_consensus([emb_a, emb_b])
    
    print(f"\n📊 Latent Consensus Score: {score:.4f}")
    
    if consensus_engine.detect_hallucination(score):
        print("⚠️ [LOW CONSENSUS] Potential Hallucination Detected. Triggering Symbolic Verifier...")
        # Here we would call agent.verifier.check_consistency() from Phase 3
    else:
        print("✅ [HIGH CONSENSUS] Reasoning Path Verified by Peer Agents.")

if __name__ == "__main__":
    run_federated_consensus_test()