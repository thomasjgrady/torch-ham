from .ham import HAM

from collections import defaultdict
from torch import Tensor
from typing import *
import torch

def gradient_descent_step(model: HAM,
              states: Mapping[str, Tensor],
              alpha: Mapping[str, float] = defaultdict(lambda: 1.0),
              pin: Set[str] = set(),
              create_graph: bool = True) -> Dict[str, Tensor]:
    """
    Performs a single step of the backprop through time algorithm. By default,
    assumes that we are in trainmode, so `create_graph` is set to true.
    """
    activations = model.activations(states)
    grads = model.grads(states, activations, create_graph=create_graph)
    states = model.step(states, grads, alpha, pin)
    return states

def gradient_descent(model: HAM,
             states: Mapping[str, Tensor],
             max_iter: int,
             alpha: Mapping[str, float] = defaultdict(lambda: 1.0),
             pin: Set[str] = set(),
             tol: float = 1e-3,
             create_graph: bool = True,
             return_history: bool = False) -> Dict[str, Tensor]:
    """
    Performs at most `max_iter` steps of the backprop through time algorithm. Stopping
    early if the states no longer change.
    """
    if return_history:
        history = [states]

    for t in range(max_iter):
        states_next = gradient_descent_step(
            model,
            states,
            alpha=alpha,
            pin=pin,
            create_graph=create_graph
        )
        with torch.no_grad():
            residuals = [states_next[name] - states[name] for name in states.keys()]
            if all([torch.all(torch.abs(r) < tol) for r in residuals]):
                break
        states = states_next
        if return_history:
            history.append(states)

    if return_history:
        return history
    return states