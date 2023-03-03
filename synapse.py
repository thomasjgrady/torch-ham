from torch import Tensor

import string
import torch
import torch.nn as nn

class Synapse(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def alignment(self, *gs) -> Tensor:
        pass

    def energy(self, *gs) -> Tensor:
        return -self.alignment(*gs)
    
class DenseSynapse(Synapse):

    def __init__(self, *args, std: float = 0.02, **kwargs) -> None:
        super().__init__()

        self.W = torch.empty(*args, **kwargs)
        nn.init.normal_(self.W, mean=0.0, std=std)
        assert self.W.dim() < 26

        chars = [f'z{c}' for c in string.ascii_lowercase[:self.W.dim()]]
        self.eqn = ','.join(chars) + f',{string.ascii_lowercase[:self.W.dim()]}->z'

    def alignment(self, *gs) -> Tensor:
        n_batch = gs[0].shape[0]
        gs = [g.view(n_batch, -1) for g in gs]
        return torch.einsum(self.eqn, *gs, self.W)
