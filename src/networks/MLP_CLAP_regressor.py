
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