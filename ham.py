from collections import defaultdict
from neurons import *
from synapses import *
from torch import Tensor

import torch
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
    
    def energy(self, states: Dict[str, Tensor], activations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        neuron_energy = torch.cat([v.unsqueeze(1) for v in self.neuron_energies(states, activations).values()], dim=1).sum(dim=1)
        synapse_energy = torch.cat([v.unsqueeze(1) for v in self.synapse_energies(activations).values()], dim=1).sum(dim=1)
        return neuron_energy + synapse_energy

    def updates(self, states: Dict[str, Tensor], activations: Dict[str, Tensor]) -> Dict[str, Tensor]:

        # Compute energy
        energy = self.energy(states, activations)

        # Concatenate states and activations into a single list to produce the update direction
        # Use sorting to ensure correct lookup later
        states_and_activations = (
            *list(map(lambda item: item[1], sorted(states.items(), key=lambda item: item[0]))),
            *list(map(lambda item: item[1], sorted(activations.items(), key=lambda item: item[0])))
        )

        # Compute gradient, saving the graph for use in backprop.
        # I.e. jvp = backward of backward
        grads = torch.autograd.grad(energy, states_and_activations, torch.ones_like(energy), create_graph=True, retain_graph=True)

        # Only care about gradient w.r.t. activations
        grad_activations = grads[-len(activations):]

        # Map each gradient activation to its corresponding key
        updates = { name: -grad_activations[i] for (i, name) in enumerate(sorted(activations.keys())) }

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