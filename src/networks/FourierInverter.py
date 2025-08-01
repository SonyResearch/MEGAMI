
import torch
import math

class FourierInverter(torch.nn.Module):
    def __init__(self, fourier_dim, original_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(fourier_dim, fourier_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(fourier_dim, fourier_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(fourier_dim // 2, original_dim)
        )
    
    def forward(self, x):
        return self.net(x)