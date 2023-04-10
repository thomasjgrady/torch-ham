from torch import Tensor
from typing import *

import numpy as np
import torch

def lagr_layer_norm(x: Tensor,
                    dim: int = -1,
                    eps: float = 1e-5,
                    gamma: Tensor = 1.0,
                    delta: Tensor = 0.0) -> Tensor:

    b = x.shape[0]
    D = np.prod([x.shape[d] for d in np.atleast_1d(dim)])
    mu = torch.mean(x, dim=-1, keepdim=True)
    xm = x - mu
    std = torch.sqrt(torch.var(xm, dim=-1, keepdim=True) + eps)
    return (D*gamma*std + delta*x).view(b, -1).sum()

def lagr_softmax(x: Tensor, dim: int = -1, beta: float = 1.0) -> Tensor:
    b = x.shape[0]
    return torch.logsumexp(x, dim=dim, keepdim=True).view(b, -1).sum(dim=1)