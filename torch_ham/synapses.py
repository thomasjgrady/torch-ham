from .lagrangians import lagr_softmax
from .utils import *
from torch import Tensor
from typing import *

import itertools
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

        super().__init__()

        self.W = nn.Parameter(torch.empty(n_in, n_hid, **kwargs))
        self.normalize = normalize
        nn.init.xavier_normal_(self.W)

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
    
class ConvSynapse(Synapse):
    """
    Convolutional synapse. If "transpose" is passed, a ConvTranspose will be
    applied to the second argument instead of a normal Conv on the first.
    """
    
    def __init__(self,
                 *args,
                 transpose: bool = False,
                 **kwargs) -> None:
        
        super().__init__()

        kernel_size = kwargs['kernel_size'] if 'kernel_size' in kwargs else args[2]
        dim = len(kernel_size)
        conv = conv_transpose_kernel_from_dim(dim) if transpose else conv_kernel_from_dim

        self.conv = conv(*args, **kwargs)
        self.f0 = nn.Identity() if transpose else conv(*args, **kwargs)
        self.f1 = conv(*args, **kwargs) if transpose else nn.Identity()

    def alignment(self, g0: Tensor, g1: Tensor) -> Tensor:
        h0, h1 = self.f0(g0), self.f1(g1)
        return torch.mul(h0.view(g0.shape[0], -1), h1.view(g1.shape[0], -1)).sum(dim=1)

class AttentionSynapse(Synapse):
    """
    Generic non-causal attention alignment.
    """
    def __init__(self,
                 n_embed_q: int,
                 n_embed_k: int,
                 n_embed: int,
                 n_heads: int,
                 lagrangian: Callable = lagr_softmax,
                 **kwargs) -> None:

        super().__init__()

        self.n_embed = n_embed
        self.n_heads = n_heads
        self.lagrangian = lagrangian

        kwargs_linear, self.kwargs_lagr, _ = filter_kwargs(nn.Linear.__init__, lagrangian, **kwargs)
        kwargs_linear.setdefault('bias', False)
        self.WQ = nn.Linear(n_embed_q, n_embed, **kwargs_linear)
        self.WK = nn.Linear(n_embed_k, n_embed, **kwargs_linear)
        self.mix_heads = nn.Linear(n_heads, 1, **kwargs_linear)

    def alignment(self, gq: Tensor, gk: Tensor) -> Tensor:

        n_batch, n_tokens_q, n_embed_q = gq.shape
        n_batch, n_tokens_k, n_embed_k = gk.shape
        n_embed = self.n_embed
        n_heads = self.n_heads

        q = self.WQ(gq).view(n_batch, n_tokens_q, n_heads, n_embed // n_heads).transpose(1, 2)
        k = self.WK(gk).view(n_batch, n_tokens_k, n_heads, n_embed // n_heads).transpose(1, 2)
        a = (q @ k.transpose(-1, -2)).transpose(1, -1)
        a = self.mix_heads(a).transpose(1, -1)
        return self.lagrangian(a, **self.kwargs_lagr)