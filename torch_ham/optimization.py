from .ham import HAM

from collections import defaultdict
from torch import Tensor
from typing import *

import numpy as np
import torch

def energy_descent_step(model: HAM,
                        states: Mapping[str, Tensor],
                        activations: Mapping[str, Tensor],
                        alpha: Mapping[str, float] = defaultdict(lambda: 1.0),
                        pin: Set[str] = set(),
                        create_graph: bool = True,
                        noise_scale:Optional[float] = None) -> Dict[str, Tensor]:
    """
    Performs a single step of descending the model's energy function. By default,
    assumes that we are in trainmode, so `create_graph` is set to true.
    """
    grads = model.grads(states, activations, create_graph=create_graph)
    if noise_scale is not None:
        for name in grads.keys():
            grads[name] = grads[name] + noise_scale*torch.randn_like(grads[name])
    states = model.step(states, grads, alpha, pin)
    return states, model.activations(states)

def energy_descent(model: HAM,
                   states: Mapping[str, Tensor],
                   activations: Mapping[str, Tensor],
                   max_iter: int = 100,
                   alpha: Mapping[str, float] = defaultdict(lambda: 1.0),
                   pin: Set[str] = set(),
                   tol: float = 1e-3,
                   create_graph: bool = True,
                   return_history: bool = False,
                   noise_scale: Optional[float] = None) -> Dict[str, Tensor]:
    """
    Performs at most `max_iter` steps of energy descent, stopping
    early if the states no longer change.
    """
    if return_history:
        history_s = [states]
        history_a = [activations]

    for t in range(max_iter):
        states_next, activations_next = energy_descent_step(
            model,
            states,
            activations,
            alpha=alpha,
            pin=pin,
            create_graph=create_graph,
            noise_scale=noise_scale
        )
        if return_history:
            history_s.append(states_next)
            history_a.append(activations_next)
        with torch.no_grad():
            residuals = [activations_next[name] - activations[name] for name in activations.keys()]
        states, activations = states_next, activations_next
        with torch.no_grad():
            if all([torch.all(torch.abs(r) < tol) for r in residuals]):
                break

    if return_history:
        return history_s, history_a
    return states, activations

def deq_fixed_point(model: HAM,
                    states: Mapping[str, Tensor],
                    activations: Mapping[str, Tensor],
                    max_iter: int = 100,
                    alpha: Mapping[str, float] = defaultdict(lambda: 1.0),
                    pin: Set[str] = set(),
                    tol: float = 1e-3,
                    noise_scale: Optional[float] = None) -> Dict[str, Tensor]:
    """
    Computes the fixed point of energy descent and uses a first-order Neumann
    approximation to compute the gradient.
    """

    # Compute the fixed point function
    states, activations = energy_descent(
        model,
        states,
        activations,
        max_iter=max_iter,
        alpha=alpha,
        pin=pin,
        tol=tol,
        create_graph=False,
        return_history=False,
        noise_scale=noise_scale
    )

    # Re-engage autograd
    states, activations = energy_descent_step(
        model,
        { name: s.detach().requires_grad_() for name, s in states.items() },
        { name: a.detach().requires_grad_() for name, a in activations.items() },
        alpha=alpha,
        pin=pin,
        create_graph=True,
        noise_scale=noise_scale
    )
    
    # Create an intermediate state for activations so that a backward hook can
    # be added
    order = list(sorted(activations.keys()))
    batch_size = activations[order[0]].shape[0]
    g_vec = torch.cat([activations[name].view(batch_size, -1) for name in order], dim=1)
    
    shapes = [activations[name].shape for name in order]
    splits = [np.prod(s[1:]) for s in shapes]
    g_out = { name: gs.reshape(s) for name, gs, s in zip(order, torch.split(g_vec, splits, dim=1), shapes) }

    # Compute single step at fixed point
    s0 = { name: s.detach().requires_grad_() for name, s in states.items() }
    g0 = { name: g.detach().requires_grad_() for name, g in activations.items() }
    s1, g1 = energy_descent_step(
        model, s0, g0, alpha=alpha, pin=pin, create_graph=True, noise_scale=noise_scale
    )

    g0_sorted = [g0[name] for name in order]
    g1_sorted = [g1[name] for name in order]

    # By the implict function theorem, we have that
    #
    #   (∂g*/∂θ)ᵀy = (∂f(g*,x)/∂θ)ᵀy(I − ∂f(g*,x)/g*)⁻ᵀy
    #
    # However, we can use a first-order Neumann approximation to the second
    # term to get
    #
    #   (∂g*/∂θ)ᵀy ≈ (∂f(g*,x)/∂θ)ᵀy(I + (∂f(g*,x)/g*)ᵀ)y
    #
    # See https://arxiv.org/pdf/2111.05177.pdf for details
    #
    # Note: This means that ALL loss terms must act on the activations,
    # NOT the model hidden states!
    def backward_hook(y: Tensor):
        batch_size = y.shape[0]
        y_sorted = [ys.reshape(s) for ys, s in zip(torch.split(y, splits, dim=1), shapes)]
        dfdg = torch.autograd.grad(g1_sorted, g0_sorted, y_sorted)
        return torch.cat([(yi + di).view(batch_size, -1) for yi, di in zip(y_sorted, dfdg)], dim=1)

    g_vec.register_hook(backward_hook)

    return states, g_out