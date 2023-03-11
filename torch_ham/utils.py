from typing import *

import torch.nn as nn

def atleast_tuple(x):
    """
    Converts x into a singleton tuple if it is not already a list.
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return tuple(x)
    else:
        return (x,)

def filter_kwargs(*funcs: Callable, **kwargs) -> List[Dict]:
    """
    Given a list of n functions `funcs` and a set of keyword arguments `kwargs`,
    split kwargs into `n+1` dicts containing all arguments for each function, along
    with the remainder.
    """
    kwarg_sets = [ set(f.__code__.co_varnames) for f in funcs ]
    kwargs_out = [ { k: v for k, v in kwargs.items() if k in kf } for kf in kwarg_sets ]
    kwargs_out.append( { k: v for k, v in kwargs.items() if all(k not in kf for kf in kwarg_sets) } )
    return kwargs_out

def conv_kernel_from_dim(dim: int):
    if dim == 1:
        return nn.Conv1d
    elif dim == 2:
        return nn.Conv2d
    elif dim == 3:
        return nn.Conv3d

def conv_transpose_kernel_from_dim(dim: int):
    if dim == 1:
        return nn.ConvTranspose1d
    elif dim == 2:
        return nn.ConvTranspose2d
    elif dim == 3:
        return nn.ConvTranspose3d