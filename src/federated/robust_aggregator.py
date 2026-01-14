# src/federated/robust_aggregator.py
import torch
import torch.nn.functional as F

class ByzantineRobustAggregator:
    def __init__(self, expected_malicious_ratio=0.3):
        self.malicious_ratio = expected_malicious_ratio

    def aggregate_with_defense(self, embeddings: list):
        """
        Implements a 'Multi-Krum' style defense for Latent Thoughts.
        We reject the vectors that are furthest from the geometric median.
        """
        if len(embeddings) < 3:
            return torch.mean(torch.stack(embeddings), dim=0)

        stacked = torch.stack(embeddings).squeeze(1) # (N, Dim)
        n = stacked.size(0)
        k = int(n * self.malicious_ratio) # Number of agents to drop
        
        # Calculate pairwise distances
        dists = torch.cdist(stacked, stacked) # (N, N)
        
        # For each agent, sum distances to its (n-k-2) nearest neighbors
        scores = []
        for i in range(n):
            # Sort distances and take the sum of the closest valid neighbors
            valid_dists, _ = torch.sort(dists[i])
            # We sum the distances to the closest (N - K - 1) neighbors
            score = valid_dists[:(n - k - 1)].sum()
            scores.append(score)
            
        # Select the agent with the minimal distance score (The Geometric Median Approx)
        best_agent_idx = torch.argmin(torch.tensor(scores))
        
        # Return the embedding of the "Most Central" (Truthful) agent
        return stacked[best_agent_idx]