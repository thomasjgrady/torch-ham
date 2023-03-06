from argparse import ArgumentParser
from ham import *

import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==== Parse CLI args ====
parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--dtype', type=torch.dtype, default=torch.float32)
parser.add_argument('--data-path', type=str, default='~/data/compass.jld2')
parser.add_argument('--num-wells', type=int, default=4)
parser.add_argument('--num-hidden', type=int, default=100)
parser.add_argument('--num-examples', type=int, default=10000)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--dt', type=float, default=0.1)
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=1e-4)
args = parser.parse_args()

# ==== Load data ====
print('Loading data... ', end='', flush=True)
device = torch.device(args.device)
dtype = args.dtype
data_path = os.path.abspath(args.data_path)
compass = torch.tensor(np.array(h5py.File(data_path, 'r')['v'])).to(dtype=dtype)
nz, ny, nx = compass.shape
print('done.', flush=True)

# ==== Define image summary net ====
class ConvBlock(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.pool = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), padding=(0, 0))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = torch.tanh(x)
        return x

image_summary_network = nn.Sequential(
    nn.Unflatten(1, (1, nz)),
    ConvBlock(1, 1, kernel_size=(16, 16), padding='same', device=device, dtype=dtype),
    ConvBlock(1, 1, kernel_size=(16, 16), padding='same', device=device, dtype=dtype),
    ConvBlock(1, 1, kernel_size=(16, 16), padding='same', device=device, dtype=dtype),
    nn.Flatten()
)

nz_out = nz // 4**(len(image_summary_network)-2)
nx_out = nx // 4**(len(image_summary_network)-2)

# ==== Build model ====
n_wells = args.num_wells
n_hidden = args.num_hidden

neurons = {
    'image': IdentityNeuron(shape=(nz, nx)),
    'hidden': SoftmaxNeuron(shape=(n_hidden,)),
    'wells': IdentityNeuron(shape=(nz, n_wells))
}

synapses = {
    's0': NetworkSynapse(
        DenseSynapse(nz_out*nx_out, n_hidden, nz*n_wells, device=device, dtype=dtype),
        image_summary_network,
        nn.Identity(),
        nn.Flatten()
    )
}

connectivity = {
    's0': ['image', 'hidden', 'wells']
}

model = HAM(neurons, synapses, connectivity)

# ==== Setup Optimizer ====
optim = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)

# ==== Training Loop ====
n_batch = args.batch_size
n_examples = args.num_examples
n_batches = n_examples // n_batch
depth = args.depth
dt = args.dt

for i in range(n_batches):
        
    # Init states
    states = model.init_states(n_batch=n_batch, device=device, dtype=dtype, requires_grad=True)
    
    # Select a random compass slice
    idxs = torch.tensor(np.random.randint(0, ny, size=n_batch), dtype=torch.long)
    
    # Select random wells
    well_idxs = torch.tensor(np.random.randint(0, nx, size=n_batch*n_wells), dtype=torch.long)
    
    # Select wells from image
    slices = torch.index_select(compass, dim=1, index=idxs).transpose(0, 1).contiguous().to(device=device)
    states['wells'] = torch.cat([torch.index_select(slices[i:i+1,:,:], dim=2, index=well_idxs[i*n_wells:(i+1)*n_wells]) for i in range(n_batch)], dim=0).requires_grad_(True)

    optim.zero_grad()

    for d in range(depth):
        activations = model.neuron_activations(states)
        updates = model.updates(states, activations)
        states = model.step(states, updates, dt, pin={'wells'})

    out = states['image']
    loss = F.mse_loss(out, slices)
    print(f'batch = {i+1:06d}/{n_batches:06d}, loss = {loss.item():2.8f}', end='\r',  flush=True)
    loss.backward()
    optim.step()