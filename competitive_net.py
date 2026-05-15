import torch
import torch.nn as nn
from tqdm import tqdm

class CompetitiveNetwork(nn.Module):
    """
    A simple unsupervised Winner-Takes-All competitive neural network.
    Uses the Kohonen learning rule to cluster input vectors.
    """
    def __init__(self, input_dim=512, num_clusters=6, learning_rate=0.1):
        super(CompetitiveNetwork, self).__init__()
        self.num_clusters = num_clusters
        self.input_dim = input_dim
        self.initial_lr = learning_rate
        
        # Initialize weights randomly. We use uniformly distributed weights and then normalize them
        # so they represent points on the unit hypersphere.
        weights = torch.randn(num_clusters, input_dim)
        self.weights = nn.Parameter(weights, requires_grad=False)
        self._normalize_weights()

    def _normalize_weights(self):
        # L2 normalization of weights
        norms = torch.norm(self.weights, p=2, dim=1, keepdim=True)
        self.weights.data = self.weights.data / (norms + 1e-8)

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        Returns the winning cluster index for each input in the batch.
        """
        # Normalize inputs
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)
        
        # Calculate distances to all cluster weights (Euclidean distance)
        # Using expanded dimensions to compute pairwise differences
        # (batch_size, 1, input_dim) - (1, num_clusters, input_dim)
        diff = x_normalized.unsqueeze(1) - self.weights.unsqueeze(0)
        distances = torch.norm(diff, p=2, dim=2)  # (batch_size, num_clusters)
        
        # Find the winning neuron (minimum distance)
        winners = torch.argmin(distances, dim=1)
        return winners

    def train_network(self, data, epochs=50):
        """
        data: (N, input_dim) tensor of all features
        Trains the network using the competitive learning rule.
        """
        N = data.shape[0]
        
        # Normalize all data once
        data_norm = torch.norm(data, p=2, dim=1, keepdim=True)
        data_normalized = data / (data_norm + 1e-8)
        
        print("Training Competitive Network...")
        for epoch in range(epochs):
            # Learning rate decay
            lr = self.initial_lr * (1.0 - epoch / epochs)
            
            # Shuffle data indices for stochastic training
            indices = torch.randperm(N)
            
            for i in tqdm(indices, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                x = data_normalized[i] # (input_dim,)
                
                # Compute distance to all weights
                diff = x.unsqueeze(0) - self.weights
                distances = torch.norm(diff, p=2, dim=1)
                
                # Winner
                winner_idx = torch.argmin(distances)
                
                # Update weights for the winner: W_winner = W_winner + lr * (X - W_winner)
                self.weights.data[winner_idx] += lr * (x - self.weights.data[winner_idx])
                
            # Re-normalize weights after epoch
            self._normalize_weights()
            
        print("Training completed.")
