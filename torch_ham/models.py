from .neurons import *
from .synapses import *
from .ham import HAM
from .utils import atleast_tuple, conv_kernel_from_dim
from typing import *

import copy

def ConditionalUNet(
    input_size: Union[int, Tuple[int]],
    condition_size: Union[int, Tuple[int]],
    kernel_size: Union[int, Tuple[int]],
    downsample: Union[int, Tuple[int]],
    depth: int,
    in_channels: int = 1,
    channel_scale: int = 1,
    **kwargs) -> None:
    """
    Creates a conditional U-Net for data generation.
    """
    
    # Make into tuples
    input_size = atleast_tuple(input_size)
    condition_size = atleast_tuple(condition_size)
    kernel_size = atleast_tuple(kernel_size)
    downsample = atleast_tuple(downsample)

    # Get dimension and corresponding conv kernel
    dim = len(input_size)
    conv = conv_kernel_from_dim(dim)

    # Get sensible defaults
    beta = kwargs.get('beta', 1.0)
    device = kwargs.get('device', torch.device('cpu'))
    dtype = kwargs.get('dtype', torch.float32)

    # Create neurons and synapses for multiscale component
    neurons = {}
    synapses = {}
    connections = {}

    current_size = copy.copy(input_size)
    current_channels = in_channels
    for d in range(depth):
        neurons[f'n{d}'] = ReluNeuron(shape=current_size)
        synapses[f's{d}'] = HopfieldSynapse(
            conv(
                current_channels,
                current_channels,
                kernel_size=kernel_size,
                padding='same',
                device=device,
                dtype=dtype
            ),
            lagrangian=lagr_softmax,
            beta=beta
        )
        connections[f's{d}'] = [f'n{d}']
        if d < depth-1:
            synapses[f's{d}_{d+1}'] = ConvSynapse(
                current_channels,
                current_channels*channel_scale,
                kernel_size=downsample,
                strides=downsample,
                device=device,
                dtype=dtype
            )
            connections[f's{d}_{d+1}'] = [f'n{d}', f'n{d+1}']

    return HAM(neurons, synapses, connections)