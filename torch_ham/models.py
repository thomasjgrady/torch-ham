from .neurons import *
from .synapses import *
from .ham import HAM
from .utils import atleast_tuple, conv_kernel_from_dim
from torch import Tensor
from typing import *

import copy

class Transpose(nn.Module):

    def __init__(self, dim0: int = -1, dim1: int = -2) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: Tensor) -> Tensor:
        return torch.transpose(x, self.dim0, self.dim1)

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
    beta = kwargs.get('beta', 0.1)
    device = kwargs.get('device', torch.device('cpu'))
    dtype = kwargs.get('dtype', torch.float32)

    # Create neurons and synapses
    neurons = {}
    synapses = {}
    connections = {}

    # Condition input
    neurons['c'] = IdentityNeuron(shape=condition_size)

    current_size = copy.copy(input_size)
    current_channels = in_channels
    for d in range(depth):
        neurons[f'n{d}'] = ReluNeuron(shape=(current_channels, *current_size))
        synapses[f's{d}'] = HopfieldSynapse(
            conv(
                current_channels,
                current_channels,
                kernel_size=kernel_size,
                padding='same',
                device=device,
                dtype=dtype
            ),
            lagrangian=lagr_relu
        )
        connections[f's{d}'] = [f'n{d}']

        synapses[f'c_n{d}'] = AttentionSynapse(
            n_embed_q=current_channels,
            n_embed_k=condition_size[-1],
            n_embed=256,
            n_heads=16,
            transform_q=nn.Sequential(
                nn.Flatten(start_dim=2),
                Transpose()
            ),
            **kwargs
        )
        connections[f'c_n{d}'] = [f'n{d}', 'c']

        if d < depth-1:
            synapses[f's{d}_{d+1}'] = ConvSynapse(
                current_channels,
                current_channels*channel_scale,
                kernel_size=downsample,
                stride=downsample,
                lagrangian=lagr_relu,
                device=device,
                dtype=dtype
            )
            connections[f's{d}_{d+1}'] = [f'n{d}', f'n{d+1}']

            current_size = tuple(s // d for s, d in zip(current_size, downsample))
            current_channels *= channel_scale

    return HAM(neurons, synapses, connections)