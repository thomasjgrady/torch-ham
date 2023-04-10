from .lagrangians import *

from torch import Tensor
from typing import *

import numpy as np
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

    def __init__(self, m: int, n: int, **kwargs) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(m, n, **kwargs))
        nn.init.kaiming_normal_(self.W)

    def alignment(self, g0: Tensor, g1: Tensor) -> Tensor:
        b = g0.shape[0]
        return torch.einsum('...m,mn,...n->...', g0, self.W, g1).view(b, -1).sum(dim=1)