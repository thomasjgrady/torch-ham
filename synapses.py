from torch import Tensor
from typing import *

import torch
import torch.nn as nn

class Synapse(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def alignment(self, *gs: Tensor) -> Tensor:
        pass

    def energy(self, *gs: Tensor) -> Tensor:
        return -self.alignment(*gs)
    
class HopfieldSynapse(Synapse):

    def __init__(self, n0: int, n1: int, n_hidden: int, beta: float = 1.0, **kwargs) -> None:
        super().__init__()
        self.beta = beta
        self.W0 = nn.Parameter(torch.empty(n0, n_hidden, **kwargs))
        self.W1 = nn.Parameter(torch.empty(n1, n_hidden, **kwargs))
        nn.init.normal_(self.W0, mean=0, std=0.02)
        nn.init.normal_(self.W1, mean=0, std=0.02)

    def alignment(self, g0: Tensor, g1: Tensor) -> Tensor:
        norm0 = torch.norm(self.W0, dim=0, keepdim=True)
        norm1 = torch.norm(self.W1, dim=0, keepdim=True)
        hidsig = g0 @ (self.W0 / norm0) + g1 @ (self.W1 / norm1)
        hidlag = 1/self.beta * torch.logsumexp(self.beta*torch.flatten(hidsig, start_dim=1), dim=-1)
        return hidlag