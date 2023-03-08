from .neurons import *
from .synapses import *

from collections import defaultdict
from torch import Tensor
from typing import *

import torch
import torch.nn as nn

class HAM(nn.Module):

    def __init__(self,
                 neurons: Mapping[str, Neuron],
                 synapses: Mapping[str, Synapse],
                 connections: Mapping[str, List[str]],
                 sensors: Mapping[str, nn.Module] = defaultdict(lambda: nn.Identity()),
                 outputs: Mapping[str, nn.Module] = defaultdict(lambda: nn.Identity())) -> None:
        
        super().__init__()

        self.neurons = nn.ModuleDict(neurons)
        self.synapses = nn.ModuleDict(synapses)
        self.connections = connections
        self.sensors = nn.ModuleDict({ name: sensors[name] for name in neurons.keys() })
        self.outputs = nn.ModuleDict({ name: outputs[name] for name in neurons.keys() })

    def init_states(self,
                    batch_size: int = 1,
                    std: DefaultDict = defaultdict(lambda: 0.02),
                    **kwargs) -> Dict[str, Tensor]:
        
        return { name: neuron.init_state(batch_size, std[name], **kwargs) for name, neuron in self.neurons.items() }

    def input(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return { name: self.sensors[name](x) for name, x in data.items() }
    
    def output(self, states: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return { name: self.outputs[name](s) for name, s in states.items() }
    
    def activations(self, states: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return { name: neuron.activations(states[name]) for name, neuron in self.neurons.items() }
    
    def neuron_energies(self, states: Dict[str, Tensor], activations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return { name: neuron.energy(states[name], activations[name]) for name, neuron in self.neurons.items() }
    
    def neuron_energy(self, states: Dict[str, Tensor], activations: Dict[str, Tensor]) -> Tensor:
        return torch.cat([e.unsqueeze(1) for e in self.neuron_energies(states, activations).values()], dim=1).sum(dim=1)
    
    def synapse_energies(self, activations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        energies = {}
        for name, synapse in self.synapses.items():
            gs = [activations[neighbor] for neighbor in self.connections[name]]
            energies[name] = synapse.energy(*gs)
        return energies
    
    def synapse_energy(self, activations: Dict[str, Tensor]) -> Tensor:
        return torch.cat([e.unsqueeze(1) for e in self.synapse_energies(activations).values()], dim=1).sum(dim=1)

    def energy(self, states: Dict[str, Tensor], activations: Dict[str, Tensor]) -> Tensor:
        return self.neuron_energy(states, activations) + self.synapse_energy(activations)
    
    def updates(self, states: Dict[str, Tensor], activations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        order = sorted(states.keys())
        acts = [activations[name] for name in order]
        energy = self.synapse_energy(activations)
        grads = torch.autograd.grad(energy, acts, torch.ones_like(energy), create_graph=True)
        return { name: -states[name] - grad for name, grad in zip(order, grads) }
    
    def step(self,
             states: Dict[str, Tensor],
             updates: Dict[str, Tensor],
             dt: float,
             tau: DefaultDict = defaultdict(lambda: 0.1),
             pin: Set[str] = set()) -> Dict[str, Tensor]:
        
        return { name: state if name in pin else state + 1/tau[name]*dt*updates[name] for name, state in states.items() }