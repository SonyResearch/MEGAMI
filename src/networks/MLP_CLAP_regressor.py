
import torch
import torch.nn as nn

class MLP_CLAP_regressor(nn.Module):
    """
    A simple MLP regressor that uses CLAP features as input.
    """

    def __init__(self, dim=512, hidden_dim=512):
        super(MLP_CLAP_regressor, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):

        emb= self.model(x)
        #l2 normalization
        return nn.functional.normalize(emb, p=2, dim=-1)
    
import torch.nn.functional as F

class EffectRemovalMLP(nn.Module):
    def __init__(self, dim=2624, hidden_dim=1024, dropout=0.1, alpha=1.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.alpha = alpha

    def forward(self, x):
        h = self.fc1(x)
        h = self.ln1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        out = self.fc2(h)
        out = self.ln2(out)
        return out
