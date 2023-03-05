from ham import *
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==== Model Parameters ====
n_batch = 100
n_image = 28*28
n_classes = 10
device = torch.device('cpu')
dtype = torch.float32

# ==== Create Model ====
neurons = {
    'input': ReluNeuron(shape=(n_image,)),
    'hidden': TanhNeuron(shape=(n_image,)),
    'label': SoftmaxNeuron(shape=(n_classes,)),
}

synapses = {
    's0': DenseSynapse(n_image, n_image, device=device, dtype=dtype),
    's1': DenseSynapse(n_image, n_classes, device=device, dtype=dtype)
}

connections = {
    's0': ['input', 'hidden'],
    's1': ['hidden', 'label']
}

model = HAM(neurons, synapses, connections)

# ==== Load Data ====
transform =transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0), (1.0))])
train_set = datasets.MNIST(os.path.expanduser('~/data'), train=True, download=True, transform=transform)
test_set = datasets.MNIST(os.path.expanduser('~/data'), train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=n_batch)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=n_batch)

# ==== Setup Optimizer ====
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==== Training Loop ====
n_epochs = 1
depth = 5
dt = 0.1

for e in range(n_epochs):
    for i, (x, y) in enumerate(train_loader):
        
        # Init states
        states = model.init_states(n_batch=n_batch, device=device, dtype=dtype, requires_grad=True)
        states['input'] = x.view(n_batch, -1).to(device=device, dtype=dtype).requires_grad_(True)

        # Move labels to device
        y = y.to(device=device, dtype=torch.long)

        optim.zero_grad()

        for d in range(depth):
            activations = model.neuron_activations(states)
            updates, energy = model.updates(states, activations, pin={'input'}, return_energy=True)
            states = model.step(states, updates, dt)
        
        logits = states['label']
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        print(f'batch = {i}, loss = {loss.item():2.8f}', end='\r',  flush=True)
        loss.backward()
        optim.step()
    break