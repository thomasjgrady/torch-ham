from torch import Tensor

import torch

def lagr_identity(x: Tensor) -> Tensor:
    """
    Lagragian of identity function.
    """
    return (0.5*x**2).flatten(start_dim=1).sum(dim=1)

def lagr_relu(x: Tensor) -> Tensor:
    """
    Lagragian of ReLU function.
    """
    return (0.5*torch.relu(x)**2).flatten(start_dim=1).sum(dim=1)

def lagr_softmax(x: Tensor, beta: float = 1.0, dim: int = -1) -> Tensor:
    """
    Lagragian of softmax function.
    """
    return 1/beta*torch.logsumexp(beta*x, dim=dim, keepdim=True) \
        .flatten(start_dim=1) \
        .sum(dim=1)