from ham import *
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_image = 28*28
n_classes = 10
device = torch.device('cuda')
dtype = torch.float32

neurons = {
    'input': ReluNeuron(shape=(n_image,)),
    'hidden': ReluNeuron(shape=(n_image,)),
    'label': SoftmaxNeuron(shape=(n_classes,)),
}

synapses = {
    's0': DenseSynapse(n_image, n_image, n_classes, device=device, dtype=dtype)
}

connections = {
    's0': ['input', 'label', 'hidden']
}

n_batch = 10
model = HAM(neurons, synapses, connections)
state = model.init_state(n_batch)
energy = model.energy(state)