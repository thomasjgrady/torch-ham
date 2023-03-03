from neuron import *
from synapse import *

from collections import defaultdict
from torch import Tensor
from typing import *

import torch
import torch.nn as nn

class HAM(nn.Module):

    def __init__(self,
                 neurons: Mapping[str, Neuron],
                 synapses: Mapping[str, Synapse],
                 connections: Mapping[str, List[str]]) -> None:
        
        super().__init__()

        self.neurons = nn.ModuleDict(neurons)
        self.synapses = nn.ModuleDict(synapses)
        self.connections = connections

    def init_state(self,
                   n_batch: int = 1,
                   std: DefaultDict = defaultdict(lambda: 0.02),
                   **kwargs) -> Dict[str, Tensor]:
        return { name: neuron.init_state(n_batch, std[name], **kwargs) for name, neuron in self.neurons.items() }
    
    def activations(self, state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return { name: neuron.activations(state[name]) for name, neuron in self.neurons.items() }
    
    def neuron_energies(self, state: Dict[str, Tensor], return_activations: bool = False) -> Dict[str, Tensor]:
        energies = { name: neuron.energy(state[name], return_activations) for name, neuron in self.neurons.items() }
        if return_activations:
            energies = { k: v[0] for (k, v) in energies.items() }
            activations = { k: v[1] for (k, v) in energies.items() }
            return energies, activations
        else:
            return energies
    
    def neuron_activations(self, state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.neuron_energies(state, return_activations=True)[1]
    
    def synapse_energies(self, activations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        energies = {}
        for name, synapse in self.synapses.items():
            gs = [activations[neighbor] for neighbor in self.connections[name]]
            energies[name] = synapse.energy(*gs)
        return energies
    
    def energy(self, state: Dict[str, Tensor], return_activations: bool = False) -> Tensor:
        neuron_energies, activations = self.neuron_energies(state, return_activations=True)
        synapse_energies = self.synapse_energies(activations)
        neuron_energy = torch.sum(torch.cat([ne.unsqueeze(-1) for ne in neuron_energies.values()], dim=-1), dim=-1, keepdim=False)
        synapse_energy = torch.sum(torch.cat([se.unsqueeze(-1) for se in synapse_energies.values()], dim=-1), dim=-1, keepdim=False)
        energy = neuron_energy + synapse_energy

        if return_activations:
            return energy, activations
        else:
            return energy