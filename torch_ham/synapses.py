from .lagrangians import lagr_softmax
from .utils import conv_kernel_from_dim
from torch import Tensor
from typing import *

import torch
import torch.nn as nn

class Synapse(nn.Module):
    """
    Base synapse class. Each synapse can compute an alignment and energy.
    """
    
    def __init__(self) -> None:
        super().__init__()

    def alignment(self, *gs: Tensor) -> Tensor:
        """
        Given activations `gs`, compute the alignment between them. States that are
        more closely aligned should return a larger value. This function must be
        differentiable.
        """
        pass

    def energy(self, *gs: Tensor) -> Tensor:
        """
        Given activations `gs`, compute the energy as the negative alignment. I.e. a
        lower energy corresponds to more closely aligned hidden states.
        """
        return -self.alignment(*gs)

class HopfieldWeight(nn.Module):
    """
    Helper module for performing the traditional Hopfield network weight multiply.
    """

    def __init__(self,
                 n_in: int,
                 n_hid: int,
                 normalize: bool = True,
                 **kwargs) -> None:

        self.W = nn.Parameter(torch.empty(n_in, n_hid, **kwargs))
        self.normalize = normalize

    def forward(self, g: Tensor) -> Tensor:
        W = self.W / torch.norm(self.W, dim=0, keepdim=True) if self.normalize else self.W
        return g @ W

class HopfieldSynapse(Synapse):
    """
    Synapse for an implict Hopfield network, where each activation `gs` is aligned
    with a shared hidden state.
    """

    def __init__(self,
                 *weights: nn.Module,
                 lagrangian: Callable = lagr_softmax,
                 **kwargs) -> None:

        super().__init__()

        self.weights = nn.ModuleList(weights)
        self.lagrangian = lagrangian
        self.lagr_kwargs = kwargs

    def alignment(self, *gs: Tensor) -> Tensor:
        hidval = torch.cat([w(g).unsqueeze(1) for w, g in zip(self.weights, gs)], dim=1).sum(dim=1)
        return self.lagrangian(hidval, **self.lagr_kwargs)

class GenericSynapse(Synapse):
    """
    Generic pairwise alignment synapse. Takes two arbitrary functions `f0` and `f1`
    and computes the alignment of `f0(g0)` and `f1(g1)` for activations `g0` and `g1`.
    """

    def __init__(self,
                 f0: nn.Module,
                 f1: nn.Module) -> None:

        super().__init__()
        self.f0 = f0
        self.f1 = f1
    
    def forward(self, g0: Tensor, g1: Tensor) -> Tensor:
        h0 = self.f0(g0)
        h1 = self.f1(g1)
        nb = h0.shape[0]
        return torch.mul(h0.view(nb, -1), h1.view(nb, -1)).sum(dim=1)

def ConvSynapse(*args, **kwargs):
    """
    Factory method for creating a pairwise alignment synapse using a
    convolutional operator.
    """
    
    kernel_size = kwargs.get('kernel_size', args[2])
    dim = len(kernel_size)
    conv = conv_kernel_from_dim(dim)

    return GenericSynapse(conv, nn.Identity())

def DenseSynapse(*args, **kwargs):
    """
    Factory method for creating a pairwise alignment synapse using a
    dense operator.
    """
    bias = kwargs.get('bias', False)
    return GenericSynapse(nn.Linear(*args, bias=bias, **kwargs))