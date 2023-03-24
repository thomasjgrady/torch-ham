from torch_ham import HAM
from torch_ham.neurons import SoftmaxNeuron, SigmoidNeuron
from torch_ham.synapses import AttentionSynapse, DenseSynapse

import torch

# ==== Define model ====

# Model parameters
vocab_size = 50_297
ctx_width = 1024
n_embed = 1024
n_heads = 64
beta = 1.0 # Inverse temperature
device = torch.device('cpu')
dtype = torch.float32

# Model hypergraph
neurons = {
    'input': SoftmaxNeuron(shape=(ctx_width, vocab_size), beta=beta),
    'output': SoftmaxNeuron(shape=(ctx_width, vocab_size), beta=beta),
    'Q': SigmoidNeuron(shape=(ctx_width, n_embed)),
    'K': SigmoidNeuron(shape=(ctx_width, n_embed))
}

synapses = {
    'encode': DenseSynapse(vocab_size, n_embed, device=device, dtype=dtype),
    'decode': DenseSynapse(vocab_size, n_embed, device=device, dtype=dtype),
    'attention': AttentionSynapse(
        n_embed_q=n_embed,
        n_embed_k=n_embed,
        n_embed=n_embed,
        n_heads=n_heads,
        ctx_width=ctx_width,
        device=device,
        dtype=dtype
    )
}

# Weight tying
synapses['decode'].W = synapses['encode'].W

connections = {
    'encode': ['input', 'K'],
    'decode': ['output', 'Q'],
    'attention': ['Q', 'K']
}

model = HAM(neurons, synapses, connections)

print(f'#params = {sum(p.numel() for p in model.parameters())}')