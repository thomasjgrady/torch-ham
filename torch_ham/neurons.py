from .lagrangians import *

from torch import Tensor
from typing import *

import numpy as np
import torch
import torch.nn as nn

class Neuron(nn.Module):

    def __init__(self, shape) -> None:
        super().__init__()
        self.shape = shape

    def init_state(self,
                   batch_size: int = 1,
                   dims: Mapping[int, int] = {},
                   std: float = 0.02,
                   **kwargs) -> Tensor:
        
        shape_out = [batch_size, *self.shape]
        for d, s in dims.items():
            shape_out[d] = s
        return std*torch.randn(*shape_out, **kwargs)
    
    def activation(self, x: Tensor) -> Tensor:
        pass

    def lagrangian(self, x: Tensor) -> Tensor:
        pass

    def energy(self, x: Tensor, g: Tensor) -> Tensor:
        b = x.shape[0]
        return torch.sum(torch.mul(x, g).view(b, -1), dim=1) - self.lagrangian(x)
    
class LayerNormNeuron(Neuron):

    def __init__(self, shape, dim: int = -1, eps: float = 1e-5, affine: bool = False, **kwargs) -> None:
        super().__init__(shape)
        self.dim = dim
        self.eps = eps
        self.affine = affine

        if affine:
            affine_shape = [1]*(len(shape)+1)
            for d in np.atleast_1d(dim):
                affine_shape[d] = shape[d-1]
            self.gamma = nn.Parameter(torch.ones(*affine_shape, **kwargs))
            self.delta = nn.Parameter(torch.zeros(*affine_shape, **kwargs))

        else:
            self.gamma = 1.0
            self.delta = 0.0

    def activation(self, x: Tensor) -> Tensor:
        mu = torch.mean(x, dim=self.dim, keepdim=True)
        xm = x - mu
        std = torch.sqrt(torch.var(x, dim=self.dim, keepdim=True) + self.eps)
        return self.gamma * xm/std + self.delta
    
    def lagrangian(self, x: Tensor) -> Tensor:
        return lagr_layer_norm(x, dim=self.dim, eps=self.eps, gamma=self.gamma, delta=self.delta)