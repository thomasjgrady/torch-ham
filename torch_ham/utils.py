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
    kwargs_out.append( { k: v for k, v in kwargs.items() if all(k not in kf) for kf in kwarg_sets } )
    return kwargs_out

def conv_kernel_from_dim(dim: int):
    if dim == 1:
        conv = nn.Conv1d(*args, **kwargs)
    elif dim == 2:
        conv = nn.Conv2d(*args, **kwargs)
    elif dim == 3:
        conv = nn.Conv3d(*args, **kwargs)