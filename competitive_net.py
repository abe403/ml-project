import torch
import torch.nn as nn
from tqdm import tqdm
class CompetitiveNetwork(nn.Module):
    def __init__(self, input_dim=512, num_clusters=6, learning_rate=0.1):
        super(CompetitiveNetwork, self).__init__()
        self.num_clusters = num_clusters
        self.input_dim = input_dim
        self.initial_lr = learning_rate
        weights = torch.randn(num_clusters, input_dim)
        self.weights = nn.Parameter(weights, requires_grad=False)
        self._normalize_weights()
    def _normalize_weights(self):
        norms = torch.norm(self.weights, p=2, dim=1, keepdim=True)
        self.weights.data = self.weights.data / (norms + 1e-8)
    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)
        diff = x_normalized.unsqueeze(1) - self.weights.unsqueeze(0)
        distances = torch.norm(diff, p=2, dim=2)
        winners = torch.argmin(distances, dim=1)
        return winners
    def train_network(self, data, epochs=50):
        N = data.shape[0]
        data_norm = torch.norm(data, p=2, dim=1, keepdim=True)
        data_normalized = data / (data_norm + 1e-8)
        print("Training Competitive Network...")
        for epoch in range(epochs):
            lr = self.initial_lr * (1.0 - epoch / epochs)
            indices = torch.randperm(N)
            for i in tqdm(indices, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                x = data_normalized[i]
                diff = x.unsqueeze(0) - self.weights
                distances = torch.norm(diff, p=2, dim=1)
                winner_idx = torch.argmin(distances)
                self.weights.data[winner_idx] += lr * (x - self.weights.data[winner_idx])
            self._normalize_weights()
        print("Training completed.")
