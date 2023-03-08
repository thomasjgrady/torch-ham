from torch import Tensor
from typing import *

import torch
import torch.nn as nn

class Synapse(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def alignment(self, *gs: Tensor) -> Tensor:
        pass

    def energy(self, *gs: Tensor) -> Tensor:
        return -self.alignment(*gs)
    
class HopfieldSynapse(Synapse):

    def __init__(self, n0: int, n1: int, n_hidden: int, beta: float = 1.0, **kwargs) -> None:
        super().__init__()
        self.beta = beta
        self.W0 = nn.Parameter(torch.empty(n0, n_hidden, **kwargs))
        self.W1 = nn.Parameter(torch.empty(n1, n_hidden, **kwargs))
        nn.init.normal_(self.W0, mean=0, std=0.02)
        nn.init.normal_(self.W1, mean=0, std=0.02)

    def alignment(self, g0: Tensor, g1: Tensor) -> Tensor:
        norm0 = torch.norm(self.W0, dim=0, keepdim=True)
        norm1 = torch.norm(self.W1, dim=0, keepdim=True)
        hidsig = g0 @ (self.W0 / norm0) + g1 @ (self.W1 / norm1)
        hidlag = 1/self.beta * torch.logsumexp(self.beta*hidsig.view(-1, hidsig.shape[-1]), dim=1)
        return hidlag.view(-1, hidlag.shape[-1])
    
class ImplicitHopfieldSynapse(Synapse):

    def __init__(self, n: int, n_hidden: int, beta: float = 1.0, **kwargs) -> None:
        super().__init__()
        self.beta = beta
        self.W = nn.Parameter(torch.empty(n, n_hidden, **kwargs))
        nn.init.normal_(self.W, mean=0, std=0.02)

    def alignment(self, g: Tensor) -> Tensor:
        norm = torch.norm(self.W, dim=0, keepdim=True)
        hidsig = g @ (self.W / norm)
        hidlag = 1/self.beta * torch.logsumexp(self.beta*hidsig, dim=-1)
        return hidlag.view(-1, hidlag.shape[-1])

class CausalSelfAttentionSynapse(Synapse):

    def __init__(self,
                 n_tokens: int,
                 n_embed: int,
                 n_heads: int,
                 bias: bool = False,
                 **kwargs) -> None:
        
        super().__init__()

        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.lift = nn.Linear(n_embed, 2*n_embed, bias=bias, **kwargs)
        self.proj = nn.Linear(n_heads, n_heads, bias=False, **kwargs)
        self.mask = torch.tril(torch.ones(n_tokens, n_tokens, device=kwargs.get('device', torch.device('cpu')), dtype=torch.bool)) \
            .view(1, 1, n_tokens, n_tokens)

    def alignment(self, g: Tensor) -> Tensor:
        
        # Extract shape info
        batch_size, n_tokens, n_embed = g.shape
        n_heads = self.n_heads

        # Project to query and key matrices
        q, k = self.lift(g).split(n_embed, dim=-1)
        q = q.view(batch_size, n_tokens, n_heads, n_embed // n_heads).transpose(1, 2)
        k = k.view(batch_size, n_tokens, n_heads, n_embed // n_heads).transpose(1, 2)

        # Compute attention matrix
        a = q @ k.transpose(-1, -2)
        a.masked_fill_(self.mask[:,:,:n_tokens,:n_tokens] == 0, float('-inf'))

        # Alignment/energy is over last dim
        out = torch.logsumexp(a, dim=-1, keepdim=False).transpose(-1, -2)

        # Linear transform over heads
        out = self.proj(out).sum(dim=2)

        return out.view(-1, out.shape[-1])