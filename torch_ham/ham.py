from .lagrangians import *
from .neurons import Neuron
from .synapses import Synapse

from collections import defaultdict
from torch import Tensor
from typing import *

import numpy as np
import torch
import torch.nn as nn

class HAM(nn.Module):

    def __init__(self,
                 neurons: Mapping[str, Neuron],
                 synapses: Mapping[str, Synapse],
                 connections: Mapping[str, List[str]],
                 transforms: Mapping[Tuple[str, str], nn.Module] = defaultdict(lambda: nn.Identity()),
                 encoders: Mapping[str, nn.Module] = defaultdict(lambda: nn.Identity()),
                 decoders: Mapping[str, nn.Module] = defaultdict(lambda: nn.Identity())) -> None:
        
        super().__init__()

        self.neurons = nn.ModuleDict(neurons)
        self.synapses = nn.ModuleDict(synapses)
        self.connections = connections
        self.transforms = nn.ModuleDict({ '#'.join(name_tuple): transform for name_tuple, transform in transforms.items() })
        self.encoders = nn.ModuleDict({ name: enc for name, enc in encoders.items() })
        self.decoders = nn.ModuleDict({ name: dec for name, dec in decoders.items() })

    def init_states(self,
                    batch_size: int = 1,
                    dims: Mapping[str, Mapping[int, int]] = defaultdict(lambda: {}),
                    std: Mapping[str, float] = defaultdict(lambda: 0.02),
                    values: Mapping[str, Tensor] = {},
                    **kwargs) -> Dict[str, Tensor]:
        
        return { name: values[name] if name in values else neuron.init_state(
            batch_size=batch_size,
            dims=dims[name],
            std=std[name],
            **kwargs
        ) for name, neuron in self.neurons.items() }
    
    def activations(self, xs: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        return { name: neuron.activation(xs[name]) for name, neuron in self.neurons.items() }
    
    def neuron_energies(self, xs: Mapping[str, Tensor], gs: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        return { name: neuron.energy(xs[name], gs[name]) for name, neuron in self.neurons.items() }
    
    def neuron_energy(self, xs: Mapping[str, Tensor], gs: Mapping[str, Tensor]) -> Tensor:
        return torch.sum(torch.cat(self.neuron_energies(xs, gs).values(), dim=1), dim=1)
    
    def synapse_energies(self, xs: Mapping[str, Tensor], gs: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        energies = {}
        for name, synapse in self.synapses.items():
            gs_neighbor = [self.transforms[f'{name}#{neighbor}'](gs[neighbor]) for neighbor in self.connections[name]]
            energies[name] = synapse.energy(*gs_neighbor)
        return energies
    
    def synapse_energy(self, gs: Mapping[str, Tensor]) -> Tensor:
        return torch.sum(torch.cat(self.synapse_energies(gs).values(), dim=1), dim=1)
    
    def energy(self, xs: Mapping[str, Tensor], gs: Mapping[str, Tensor]) -> Tensor:
        return self.neuron_energy(xs) + self.synapse_energy(gs)
    
    def grads(self, xs: Mapping[str, Tensor], gs: Mapping[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        order = list(sorted(xs.keys()))
        E = self.synapse_energy(gs)
        gs_sorted = [gs[name] for name in order]
        dEdg = torch.autograd.grad(outputs=E, inputs=gs_sorted, grad_outputs=torch.ones_like(E), **kwargs)
        return { name: xs[name] + grad for name, grad in zip(order, dEdg) }
    
    def energy_descent_step(self,
                            xs: Mapping[str, Tensor],
                            gs: Mapping[str, Tensor],
                            alpha: Mapping[str, float] = defaultdict(lambda: 1.0),
                            pin: Set[str] = {},
                            **autograd_kwargs) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        
        grads = self.grads(xs, gs, **autograd_kwargs)
        xs_next = { name: x if name in pin else x - alpha[name]*grads[name] for name, x in xs.items() }
        gs_next = self.activations(xs_next)
        return xs_next, gs_next
    
    def energy_descent(self,
                       xs: Mapping[str, Tensor],
                       gs: Mapping[str, Tensor],
                       alpha: Mapping[str, float] = defaultdict(lambda: 1.0),
                       pin: Set[str] = {},
                       max_iter: int = 100,
                       tol: float = 1e-3,
                       **autograd_kwargs) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        
        for t in range(max_iter):
            xs_next, gs_next = self.energy_descent_step(xs, gs, alpha=alpha, pin=pin, **autograd_kwargs)
            with torch.no_grad():
                residuals = { name: torch.norm((gs_next[name] - gs[name]).view(gs[name].shape[0], -1), dim=1) for name in gs.keys() }
                xs = xs_next
                gs = gs_next
                if all(torch.all(r <= tol) for r in residuals):
                    break

        return xs, gs
    
    def deq_fixed_point(self,
                        xs: Mapping[str, Tensor],
                        gs: Mapping[str, Tensor],
                        alpha: Mapping[str, float] = defaultdict(lambda: 1.0),
                        pin: Set[str] = {},
                        max_iter: int = 100,
                        tol: float = 1e-3) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        
        xs_pinned = { name: x for name, x in xs.items() if name in pin }
        gs_pinned = { name: g for name, g in gs.items() if name in pin }

        xs_fixed, gs_fixed = self.energy_descent(
            xs,
            gs,
            alpha=alpha,
            pin=pin,
            max_iter=max_iter,
            tol=tol,
            create_graph=False
        )

        xs = { name: xs_pinned[name] if name in pin else xs_fixed[name].detach().requires_grad_() for name in xs.keys() }
        gs = { name: gs_pinned[name] if name in pin else gs_fixed[name].detach().requires_grad_() for name in gs.keys() }
        xs, gs = self.energy_descent_step(xs, gs, alpha=alpha, pin=pin, create_graph=True)

        order = [name for name in sorted(xs.keys()) if name not in pin]
        xs_vec = torch.cat([xs[name].view(xs[name].shape[0], -1) for name in order], dim=1)
        gs_vec = torch.cat([gs[name].view(gs[name].shape[0], -1) for name in order], dim=1)
        hook_target = torch.cat((xs_vec, gs_vec), dim=1)

        xs_vec_len = xs_vec.shape[1]
        gs_vec_len = gs_vec.shape[1]
        xs_shapes = [xs[name].shape for name in order]
        gs_shapes = [gs[name].shape for name in order]
        xs_strides = [np.prod(s[1:]) for s in xs_shapes]
        gs_strides = [np.prod(s[1:]) for s in gs_shapes]

        xs_vec, gs_vec = hook_target.split([xs_vec_len, gs_vec_len], dim=1)
        xs_unpinned = { name: x.view(s) for name, x, s in zip(order, xs_vec.split(xs_strides, dim=1), xs_shapes) }
        gs_unpinned = { name: g.view(s) for name, g, s in zip(order, gs_vec.split(gs_strides, dim=1), gs_shapes) }

        xs_out = { name: xs_pinned[name] if name in pin else xs_unpinned[name] for name in xs.keys() }
        gs_out = { name: gs_pinned[name] if name in pin else gs_unpinned[name] for name in gs.keys() }

        return xs_out, gs_out