from collections import defaultdict
from neurons import *
from synapses import *
from torch import Tensor

import itertools
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn

class HAM(nn.Module):

    def __init__(self,
                 neurons: Dict[str, Neuron],
                 synapses: Dict[str, Synapse],
                 connectivity: Dict[str, List[str]]) -> None:
        
        super().__init__()

        self.neurons = nn.ModuleDict(neurons)
        self.synapses = nn.ModuleDict(synapses)
        self.connectivity = connectivity

    def neuron_activations(self, states: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return { name: neuron.activations(states[name]) for name, neuron in self.neurons.items() }
    
    def neuron_energies(self, states: Dict[str, Tensor], activations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return { name: neuron.energy(states[name], activations[name]) for name, neuron in self.neurons.items() }
    
    def synapse_energies(self, activations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        energies = {}
        for name, synapse in self.synapses.items():
            gs = [activations[neighbor] for neighbor in self.connectivity[name]]
            energies[name] = synapse.energy(*gs)
        return energies

    def neuron_energy(self, states: Dict[str, Tensor], activations: Dict[str, Tensor]) -> Tensor:
        return torch.cat([v.unsqueeze(1) for v in self.neuron_energies(states, activations).values()], dim=1).sum(dim=1)

    def synapse_energy(self, activations: Dict[str, Tensor]) -> Tensor:
        return torch.cat([v.unsqueeze(1) for v in self.synapse_energies(activations).values()], dim=1).sum(dim=1)

    def energy(self, states: Dict[str, Tensor], activations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.neuron_energy(states, activations) + self.synapse_energy(activations)

    def updates(self,
                states: Dict[str, Tensor],
                activations: Dict[str, Tensor],
                return_energy: bool = False) -> Dict[str, Tensor]:

        # Compute energy gradient
        energy = self.energy(states, activations)
        order = sorted(activations.keys())
        acts = [activations[name] for name in order]
        dEdg = torch.autograd.grad(energy, acts, torch.ones_like(energy), create_graph=True)

        # Map each gradient activation to its corresponding key
        updates = { name: -grad for name, grad in zip(order, dEdg) }

        if return_energy:
            return updates, energy
        return updates
    
    def step(self,
             states: Dict[str, Tensor],
             updates: Dict[str, Tensor],
             dt: float,
             tau: DefaultDict = defaultdict(lambda: 0.1),
             pin: Set[str] = set()) -> Dict[str, Tensor]:
        
        return { name: state if name in pin else state + dt/tau[name]*updates[name] for name, state in states.items() }
    
    def init_states(self,
                    n_batch: int = 1,
                    std: DefaultDict = defaultdict(lambda: 0.02),
                    **kwargs) -> Dict[str, Tensor]:
        
        return { name: neuron.init_state(n_batch, std[name], **kwargs) for name, neuron in self.neurons.items() }