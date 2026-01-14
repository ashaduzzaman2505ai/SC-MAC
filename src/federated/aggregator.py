import torch
import torch.nn.functional as F

class FederatedConsensus:
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def compute_consensus(self, agent_embeddings: list):
        """
        Calculates the agreement between N agents using cosine similarity.
        High similarity in latent space suggests factual alignment.
        """
        if not agent_embeddings:
            return 0.0
            
        # Stack embeddings (N, Hidden_Dim)
        stacked = torch.stack(agent_embeddings).squeeze(1)
        
        # Compute Pairwise Cosine Similarity
        sim_matrix = F.cosine_similarity(stacked.unsqueeze(0), stacked.unsqueeze(1), dim=-1)
        
        # Mask diagonal (self-similarity) and average
        mask = torch.eye(sim_matrix.size(0)).to(sim_matrix.device)
        consensus_score = (sim_matrix * (1 - mask)).sum() / (sim_matrix.size(0) * (sim_matrix.size(0) - 1))
        
        return consensus_score.item()

    def detect_hallucination(self, score: float):
        """Decision gate for the Agentic Loop."""
        return score < self.threshold