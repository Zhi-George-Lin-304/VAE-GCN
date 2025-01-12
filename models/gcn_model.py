import torch
import torch.nn as nn
from models.gcn_layer import GCNLayer

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn4 = GCNLayer(hidden_dim, hidden_dim)

    def forward(self, A_normalized, X, num_atoms):
        H = self.gcn1(A_normalized, X)
        H = self.gcn2(A_normalized, H)
        H = self.gcn3(A_normalized, H)
        H = self.gcn4(A_normalized, H)

        mask = torch.arange(H.size(1), device=H.device).expand(H.size(0), -1) < num_atoms.unsqueeze(1)
        H_masked = H * mask.unsqueeze(2)
        embeddings = H_masked.sum(dim=1) / num_atoms.unsqueeze(1).float()

        return embeddings

