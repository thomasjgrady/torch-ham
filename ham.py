from collections import defaultdict
from neurons import *
from synapses import *
from torch import Tensor

import torch
import torch.nn as nn

class StateUpdate(torch.autograd.Function):

    @staticmethod
    def forward(self,
                ctx,
                activations: Dict[str, Tensor],
                energy: Dict[str, Tensor]) -> Dict[str, Tensor]:

        order = list(sorted(name for name in in activations.keys()))
        activations = [activations[name] for name in ctx.order]
        grads = torch.autograd.grad(energy, activations, torch.ones_like(energy))[1:]

        

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

    def updates(self,
                states: Dict[str, Tensor],
                activations: Dict[str, Tensor],
                return_energy: bool = False,
                pin: Set[str] = set()) -> Dict[str, Tensor]:

        # Compute energy
        energy = self.energy(states, activations)

        # Get all activations that are not pinned and gradient calculation
        unpinned = list(sorted(name for name, activation in activations.items() if name not in pin and activation.requires_grad))
        pinned = { name for name in activations.keys() if name not in unpinned }
        activations_unpinned = tuple(activations[name] for name in unpinned)

        # Compute gradient, saving the graph for use in backprop.
        # I.e. jvp = backward of backward
        grads = torch.autograd.grad(energy, activations_unpinned, torch.ones_like(energy), create_graph=True)

        # Map each gradient activation to its corresponding key
        updates = { name: -grad for name, grad in zip(unpinned, grads) }

        if return_energy:
            return updates, energy
        return updates
    
    def step(self,
             states: Dict[str, Tensor],
             updates: Dict[str, Tensor],
             dt: float,
             tau: DefaultDict = defaultdict(lambda: 0.1)) -> Dict[str, Tensor]:
        
        return { name: state + dt/tau[name]*updates[name] if name in updates else state for name, state in states.items() }
    
    def init_states(self,
                    n_batch: int = 1,
                    std: DefaultDict = defaultdict(lambda: 0.02),
                    **kwargs) -> Dict[str, Tensor]:
        
        return { name: neuron.init_state(n_batch, std[name], **kwargs) for name, neuron in self.neurons.items() }