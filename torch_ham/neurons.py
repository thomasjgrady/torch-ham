from torch import Tensor
from typing import *

import torch
import torch.nn as nn

class Neuron(nn.Module):

    def __init__(self, shape: Tuple[int]) -> None:
        super().__init__()
        self.shape = shape

    def activations(self, x: Tensor) -> Tensor:
        pass

    def lagrangian(self, x: Tensor) -> Tensor:
        pass

    def energy(self, x: Tensor, g: Tensor) -> Tensor:
        x = torch.flatten(x, start_dim=1)
        g = torch.flatten(g, start_dim=1)
        return torch.multiply(x, g).sum(dim=1) - self.lagrangian(x)

    def init_state(self, batch_size: int = 1, std: float = 0.02, **kwargs) -> Tensor:
        return std*torch.randn(batch_size, *self.shape, **kwargs)

class SoftmaxNeuron(Neuron):

    def __init__(self, shape: Tuple[int], beta: float = 1.0) -> None:
        super().__init__(shape)
        self.beta = beta

    def activations(self, x: Tensor) -> Tensor:
        return torch.softmax(self.beta*x, dim=-1)

    def lagrangian(self, x: Tensor) -> Tensor:
        return 1/self.beta * torch.logsumexp(self.beta*x.view(x.shape[0], -1), dim=-1)
    
class SphericalNeuron(Neuron):

    def __init__(self, shape: Tuple[int], eps=1e-6) -> None:
        super().__init__(shape)
        self.eps = eps

    def activations(self, x: Tensor) -> Tensor:
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x = x / (torch.norm(x, dim=-1, keepdim=True) + self.eps)
        return x
    
    def lagrangian(self, x: Tensor) -> Tensor:
        return torch.norm(x.view(x.shape[0], -1), dim=-1)