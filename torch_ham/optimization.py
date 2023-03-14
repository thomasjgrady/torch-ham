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

class DEQFixedPoint(torch.autograd.Function):
    """
    Computes fixed point and uses a first-order Neumann approximation to compute
    implicit derivative.
    """

    @staticmethod
    def forward(ctx: Any,
                model: HAM,
                order: Iterable[str],
                max_iter: int,
                alpha: Mapping[str, float],
                pin: Set[str],
                tol: float,
                *states: Tensor) -> Tuple[Tensor]:
        
        # Autograd functions cannot take dicts as input
        states = { name: s for name, s in zip(order, states) }        

        # Compute fixed point
        with torch.enable_grad():
            z_star = gradient_descent(
                model,
                states,
                max_iter=max_iter,
                alpha=alpha,
                pin=pin,
                tol=tol,
                create_graph=False
            )

            # Engage autograd
            z0 = { name: z.detach().clone().requires_grad_() for name, z in z_star.items() }
            f0 = gradient_descent_step(model, states, alpha=alpha, pin=pin, create_graph=True)

        ctx.order = list(sorted(z0.keys()))
        ctx.z0_sorted = [z0[name] for name in ctx.order]
        ctx.f0_sorted = [f0[name] for name in ctx.order]

        return tuple([z_star[name] for name in order])
    
    @staticmethod
    def backward(ctx: Any, *y: Tensor) -> Tuple[Tensor]:

        # Implicit derivative gives JVP
        # 
        #   (∂z*/∂θ)ᵀy = (∂f(z*,θ)/∂θ)ᵀ(I − ∂f(z*,θ)/∂z*)⁻ᵀy
        # 
        # However, we can approximate (I − ∂f(z*,θ)/∂z*)⁻ᵀy by a first-order
        # Neumann approximation as simply
        #
        #   (I − ∂f(z*,θ)/∂z*)⁻ᵀy ≈ (∂f(z*,θ)/∂z*)ᵀy
        #
        with torch.enable_grad():
            # `z0_sorted` is detached from the rest of the graph, so derivatives will
            # not propagate beyond it in `autograd.backward`
            torch.autograd.backward(ctx.f0_sorted, y, retain_graph=True)
        
        grads = [None, None]
        grads.extend([z.grad for z in ctx.z0_sorted])
        grads.extend([None, None, None, None, None])
        return tuple(grads)
    
def deq_fixed_point(model: HAM,
                    states: Dict[str, Tensor],
                    max_iter: int = 100,
                    alpha: Mapping[str, float] = defaultdict(lambda: 1.0),
                    pin: Set[str] = set(),
                    tol: float = 1e-3) -> Dict[str, Tensor]:
    
    order = list(sorted(states.keys()))
    states_sorted = [states[name] for name in order]
    fixed_points_sorted = DEQFixedPoint.apply(
        model,
        order,
        max_iter,
        alpha,
        pin,
        tol,
        *states_sorted
    )
    return { name: x for name, x in zip(order, fixed_points_sorted) }