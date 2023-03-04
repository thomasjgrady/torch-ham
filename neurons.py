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
        l = self.lagrangian(x)
        return torch.mul(x.view(x.shape[0], -1), g.view(g.shape[0], -1)).sum(dim=1, keepdim=False) - l
    
    def init_state(self, n_batch: int = 1, std: float = 0.02, **kwargs) -> Tensor:
        return std*torch.randn(n_batch, *self.shape, **kwargs)
    
class ReluNeuron(Neuron):

    def __init__(self, shape: Tuple[int]) -> None:
        super().__init__(shape)

    def activations(self, x: Tensor) -> Tensor:
        return torch.relu(x)
    
    def lagrangian(self, x: Tensor) -> Tensor:
        return (0.5*torch.relu(x.view(x.shape[0], -1))**2).sum(dim=1, keepdim=False)
    
class SoftmaxNeuron(Neuron):

    def __init__(self, shape: Tuple[int]) -> None:
        super().__init__(shape)

    def activations(self, x: Tensor) -> Tensor:
        return torch.softmax(x, dim=-1)
    
    def lagrangian(self, x: Tensor) -> Tensor:
        return torch.logsumexp(x.view(x.shape[0], -1), dim=1)
    
class TanhNeuron(Neuron):

    def __init__(self, shape: Tuple[int]) -> None:
        super().__init__(shape)

    def activations(self, x: Tensor) -> Tensor:
        return torch.tanh(x)
    
    def lagrangian(self, x: Tensor) -> Tensor:
        return torch.log(torch.cosh(x.view(x.shape[0], -1))).sum(dim=1, keepdim=False)