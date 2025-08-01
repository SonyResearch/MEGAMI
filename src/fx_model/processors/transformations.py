import torch
import torch.nn as nn
from typing import Union, Optional

class WrappedPositive(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period

    def forward(self, x):
        return x.abs() % self.period

    def right_inverse(self, y):
        return y


class SmoothingCoef(nn.Module):
    def forward(self, x):
        return x.sigmoid()

    def right_inverse(self, y):
        return (y / (1 - y)).log()

class Identity(nn.Module):
    def forward(self, x):
        return x

    def right_inverse(self, y):
        return y

class MinMax(nn.Module):
    def __init__(self, min=0.0, max: Union[float, torch.Tensor] = 1.0):
        super().__init__()
        if isinstance(min, torch.Tensor):
            self.register_buffer("min", min, persistent=False)
        else:
            self.min = min

        if isinstance(max, torch.Tensor):
            self.register_buffer("max", max, persistent=False)
        else:
            self.max = max

        self._m = SmoothingCoef()

    def forward(self, x):
        return self._m(x) * (self.max - self.min) + self.min

    def right_inverse(self, y):
        return self._m.right_inverse((y - self.min) / (self.max - self.min))
    

class UniLossLess(nn.Module):
    def forward(self, x):
        if x.dim() ==2:
            tri = x.triu(1)
            return torch.linalg.matrix_exp(tri - tri.T)
        elif x.dim() == 3:
            B, N,N= x.shape
            out= torch.zeros_like(x)
            for i in range(B):
                out[i]=torch.linalg.matrix_exp(x[i].triu(1) - x[i].triu(1).T)
            return out


    def right_inverse(self, y):
        #copilot generated this...
        #tri = y.triu(1)
        #return tri + tri.T - torch.diag(tri.diagonal()) 
        raise NotImplementedError("UniLossLess right_inverse is not implemented, please implement it if you need it.")

