from torch import Tensor
from typing import *

import string
import torch
import torch.nn as nn

class Synapse(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def alignment(self, *gs: Tensor) -> Tensor:
        pass

    def energy(self, *gs: Tensor) -> Tensor:
        return -self.alignment(*gs)
    
class DenseSynapse(Synapse):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(*args, **kwargs))
        nn.init.xavier_normal_(self.W)

        s = self.W.shape
        d = len(s)
        self.eqn = f"{','.join(f'z{c}' for c in string.ascii_lowercase[:d])},{string.ascii_lowercase[:d]}->z"

    def alignment(self, *gs: Tensor) -> Tensor:
        gs = [g.view(g.shape[0], -1) for g in gs]
        return torch.einsum(self.eqn, *gs, self.W)

    def forward(self, *gs):
        return self.energy(*gs)