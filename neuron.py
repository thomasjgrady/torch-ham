from torch import Tensor

import torch
import torch.nn as nn

class Neuron(nn.Module):

    def __init__(self, shape) -> None:
        super().__init__()
        self.shape = shape

    def activations(self, x: Tensor) -> Tensor:
        pass

    def lagrangian(self, x: Tensor) -> Tensor:
        pass

    def energy(self, x: Tensor, return_activations: bool = False) -> Tensor:
        n_batch = x.shape[0]
        g = self.activations(x).view(n_batch, -1)
        l = self.lagrangian(x)
        e = torch.mul(x, g).sum(dim=1, keepdim=False) - l

        if return_activations:
            return e, g
        else:
            return e
        
    def init_state(self, n_batch: int = 1, std: float = 0.02, **kwargs) -> Tensor:
        return std*torch.randn(n_batch, *self.shape, **kwargs)
        
class SoftmaxNeuron(Neuron):

    def __init__(self, shape, dim=-1) -> None:
        super().__init__(shape)
        self.dim = dim

    def activations(self, x: Tensor) -> Tensor:
        return torch.softmax(x, dim=self.dim)
    
    def lagrangian(self, x: Tensor) -> Tensor:
        return torch.logsumexp(x, dim=self.dim, keepdim=False).view(x.shape[0])
    
class ReluNeuron(Neuron):

    def __init__(self, shape, dim=-1) -> None:
        super().__init__(shape)
        self.dim = dim

    def activations(self, x: Tensor) -> Tensor:
        return torch.relu(x)
    
    def lagrangian(self, x: Tensor) -> Tensor:
        n_batch = x.shape[0]
        r = 0.5*torch.relu(x)**2
        return torch.sum(r.view(n_batch, -1), dim=self.dim, keepdim=False)
