from .neurons import Neuron
from .synapses import Synapse
from collections import defaultdict
from torch import Tensor
from typing import *

import torch
import torch.nn as nn

class HAM(nn.Module):
    """
    Hierarchical associative memory network.
    """

    def __init__(self,
                 neurons: Mapping[str, Neuron],
                 synapses: Mapping[str, Synapse],
                 connections: Mapping[str, List[str]],
                 transforms: Mapping[Tuple[str, str], nn.Module] = defaultdict(lambda: nn.Identity()),
                 sensors: Mapping[str, nn.Module] = defaultdict(lambda: nn.Identity()),
                 outputs: Mapping[str, nn.Module] = defaultdict(lambda: nn.Identity())) -> None:

        super().__init__()

        self.neurons = nn.ModuleDict(neurons)
        self.synapses = nn.ModuleDict(synapses)
        self.connections = connections
        self.sensors = nn.ModuleDict({ name: sensors[name] for name in neurons.keys() })
        self.outputs = nn.ModuleDict({ name: outputs[name] for name in neurons.keys() })

        self.transforms = nn.ModuleDict()
        for name in synapses.keys():
            for neighbor in connections[name]:
                self.transforms[f'{name}_{neighbor}'] = transforms[(name, neighbor)]

    def init_states(self,
                    mean: Mapping[str, float] = defaultdict(lambda: 0.0),
                    std: Mapping[str, float] = defaultdict(lambda: 0.02),
                    batch_size: int = 1,
                    exclude: Set[str] = set(),
                    **kwargs) -> None:
        """
        Initializes neuron states using the given means and standard deviations
        for each neuron. Any identifiers present in `exclude` will be initialized
        to `None` (useful to avoid extra memory allocation in training loops where
        a particular state is manually initialized).
        """
        return { name: None if name in exclude else neuron.init_state(
            mean[name],
            std[name],
            batch_size=batch_size,
            **kwargs)
        for name, neuron in self.neurons.items() }

    def activations(self, states: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """
        Computes the activation of model neurons given corresponding `states`.
        """
        return { name: neuron.activations(states[name]) for name, neuron in self.neurons.items() }

    def neuron_energies(self, states: Mapping[str, Tensor], activations: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """
        Computes the energy of model neurons given corresponding `states`
        and `activations`.
        """
        return { name: neuron.energy(states[name], activations[name]) for name, neuron in self.neurons.items() }
    
    def neuron_energy(self, states: Mapping[str, Tensor], activations: Mapping[str, Tensor]) -> Tensor:
        """
        Computes the aggregate energy of all neurons in the model using the given
        `states` and `activations`. Returns a one-dimensional tensor of length
        `batch_size`.
        """
        return torch.cat([v.unsqueeze(1) for v in self.neuron_energies(states, activations).values()], dim=1).sum(dim=1)

    def synapse_energies(self, activations: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the energy of model synapses given corresponding `activations`.
        """
        energies = {}
        for name, synapse in self.synapses.items():
            gs = [self.transforms[f'{name}_{neighbor}'](activations[neighbor]) for neighbor in self.connections[name]]
            energies[name] = synapse.energy(*gs)
        return energies

    def synapse_energy(self, activations: Mapping[str, Tensor]) -> Tensor:
        """
        Computes the aggregate energy of all synapses in the model using the given
        `activations`. Returns a one-dimensional tensor of length `batch_size`.
        """
        return torch.cat([v.unsqueeze(1) for v in self.synapse_energies(activations).values()], dim=1).sum(dim=1)

    def energy(self, states: Mapping[str, Tensor], activations: Mapping[str, Tensor]) -> Tensor:
        """
        Computes the aggregate energy of the model using the given `states`
        and `activations`. Returns a one-dimensional tensor of length `batch_size`.
        """
        return self.neuron_energy(states, activations) + self.synapse_energy(activations)

    def grads(self, states: Mapping[str, Tensor], activations: Mapping[str, Tensor], create_graph=True) -> Tensor:
        """
        Computes the gradient of model energy w.r.t. `activations` using the given
        `states` and corresponding `activations`.
        """

        # We can employ a mathematical trick to only have to compute synapse energy
        synapse_energy = self.synapse_energy(activations)
        order = sorted(states.keys())
        acts = [activations[name] for name in order]
        grads = torch.autograd.grad(synapse_energy, acts, torch.ones_like(synapse_energy), create_graph=create_graph)
        return { name: states[name] + g for name, g in zip(order, grads) }

    def step(self,
             states: Mapping[str, Tensor],
             grads: Mapping[str, Tensor],
             alpha: Mapping[str, float] = defaultdict(lambda: 1.0),
             pin: Set[str] = set()) -> Dict[str, Tensor]:
        """
        Given a set of `states` and corresponding model energy gradients `grads`,
        step `states` towards a direction of minimal energy with corresponding step
        size `alpha`. If a state has identifier in `pin`, it will not be updated.
        """
        return { name: state if name in pin else state - alpha*grads[name] for name, state in states.items() }