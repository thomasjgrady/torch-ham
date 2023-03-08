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
        x = x.view(-1, x.shape[-1])
        g = g.view(-1, x.shape[-1])
        return torch.multiply(x, g).sum(dim=-1) - self.lagrangian(x)

    def init_state(self, batch_size: int = 1, std: float = 0.02, **kwargs) -> Tensor:
        return std*torch.randn(batch_size, *self.shape, **kwargs)

class ReluNeuron(Neuron):

    def __init__(self, shape: Tuple[int]) -> None:
        super().__init__(shape)

    def activations(self, x: Tensor) -> Tensor:
        return torch.relu(x)

    def lagrangian(self, x: Tensor) -> Tensor:
        return (0.5*torch.relu(x.view(-1, x.shape[-1]))**2).sum(dim=1)

class SoftmaxNeuron(Neuron):

    def __init__(self, shape: Tuple[int], beta: float = 1.0) -> None:
        super().__init__(shape)
        self.beta = beta

    def activations(self, x: Tensor) -> Tensor:
        return torch.softmax(self.beta*x, dim=-1)

    def lagrangian(self, x: Tensor) -> Tensor:
        return 1/self.beta * torch.logsumexp(self.beta*x.view(-1, x.shape[-1]), dim=1)
    
class SphericalNeuron(Neuron):

    def __init__(self, shape: Tuple[int], eps=1e-6) -> None:
        super().__init__(shape)
        self.eps = eps

    def activations(self, x: Tensor) -> Tensor:
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x = x / (torch.norm(x, dim=-1, keepdim=True) + self.eps)
        return x
    
    def lagrangian(self, x: Tensor) -> Tensor:
        return torch.norm(x.view(-1, x.shape[-1]), dim=1)
    
class LayerNormNeuron(Neuron):

    def __init__(self, shape: Tuple[int], eps=1e-6, **kwargs) -> None:
        super().__init__(shape)

        self.eps = eps

    def activations(self, x: Tensor) -> Tensor:
        mu = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return (x-mu)/(std + self.eps)
    
    def lagrangian(self, x: Tensor) -> Tensor:
        x = x.view(-1, x.shape[-1])
        D = x.shape[1]
        mu = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(mu, dim=1, keepdim=True)
        return D*std + x.sum(dim=1)