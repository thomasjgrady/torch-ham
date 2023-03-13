from .lagrangians import *
from torch import Tensor
from typing import *

import torch
import torch.nn as nn

class Neuron(nn.Module):
    """
    Base class representing a neuron. Each neuron has an associated state, and
    can compute activations, lagrangians, and associated energy.
    """

    def __init__(self, shape: Union[int, Tuple[int]]) -> None:
        super().__init__()
        self.shape = shape

    def init_state(self, mean=0.0, std=0.02, batch_size: int = 1, **kwargs) -> Tensor:
        """
        Initialize the neuron state with the given mean and standard deviation.
        """
        return std*torch.randn(batch_size, *self.shape, **kwargs) + mean

    def activations(self, x: Tensor) -> Tensor:
        """
        Compute the activations on the given state `x`.
        """
        pass

    def lagrangian(self, x: Tensor) -> Tensor:
        """
        Compute the Lagrangian of the given state `x`.
        """
        pass

    def energy(self, x: Tensor, g: Tensor) -> Tensor:
        """
        Compute the energy associated with state `x` and corresponding activations `g`.
        """
        return torch.mul(x.flatten(start_dim=1), g.flatten(start_dim=1)).sum(dim=1) - self.lagrangian(x)

class IdentityNeuron(Neuron):
    """
    Neuron representing identity transformation.
    """

    def __init__(self, shape: Union[int, Tuple[int]]) -> None:
        super().__init__(shape)

    def activations(self, x: Tensor) -> Tensor:
        return x

    def lagrangian(self, x: Tensor) -> Tensor:
        return lagr_identity(x)

class ReluNeuron(Neuron):
    """
    Neuron representing rectified linear unit (ReLU) transformation.
    """

    def __init__(self, shape: Union[int, Tuple[int]]) -> None:
        super().__init__(shape)

    def activations(self, x: Tensor) -> Tensor:
        return torch.relu(x)
    
    def lagrangian(self, x: Tensor) -> Tensor:
        return lagr_relu(x)
    
class SigmoidNeuron(Neuron):
    """
    Neuron representing sigmoid transformation with scale `scale` and
    temperature `beta`
    """

    def __init__(self, shape: Union[int, Tuple[int]], scale: float = 1.0, beta: float = 1.0) -> None:
        super().__init__(shape)
        self.scale = scale
        self.beta = beta

    def activations(self, x: Tensor) -> Tensor:
        return self.scale/(1+torch.exp(-self.beta*x))

    def lagrangian(self, x: Tensor) -> Tensor:
        return lagr_sigmoid(x, self.beta, self.scale)

class SoftmaxNeuron(Neuron):
    """
    Neuron representing softmax transformation over a given dimension `dim` with
    inverse temperature `beta`.
    """

    def __init__(self, shape: Union[int, Tuple[int]], beta: float = 1.0, dim: int = -1) -> None:
        super().__init__(shape)
        self.beta = beta
        self.dim = dim

    def activations(self, x: Tensor) -> Tensor:
        return torch.softmax(self.beta*x, dim=self.dim)

    def lagrangian(self, x: Tensor) -> Tensor:
        return lagr_softmax(x, self.beta, self.dim)
    
class SphericalNormNeuron(Neuron):
    """
    Neuron representing spherical norm transformation.
    """
    def __init__(self, shape: Union[int, Tuple[int]], dim: int = -1, eps: float = 1e-7) -> None:
        super().__init__(shape)
        self.dim = dim
        self.eps = eps

    def activations(self, x: Tensor) -> Tensor:
        x = x - torch.mean(x, dim=self.dim, keepdim=True)
        x = x / (torch.norm(x, dim=self.dim, keepdim=True) + self.eps)
        return x
    
    def lagrangian(self, x: Tensor) -> Tensor:
        return lagr_spherical_norm(x, dim=self.dim, eps=self.eps)