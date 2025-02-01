import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, output_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, A_normalized, H):
        HW = self.linear(H)
        HW_plus_B = HW + self.bias
        output = torch.matmul(A_normalized, HW_plus_B)
        return self.leaky_relu(output)
