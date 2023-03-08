from torch_ham.neurons import LayerNormNeuron
from torch_ham.synapses import CausalSelfAttentionSynapse, ImplicitHopfieldSynapse
from torch_ham import HAM

import matplotlib.pyplot as plt
import numpy as np
import os
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

data_path = os.path.expanduser('~/data/wikitext-103/wiki.train.tokens_bpe.npy')

print('Loading training data... ', end='', flush=True)
data = np.memmap(data_path, dtype=np.uint16, mode='r')
print('done.', flush=True)

tokenizer = tiktoken.get_encoding('gpt2')

n_tokens = 512
n_embed = 512
n_vocab = tokenizer.max_token_value
n_heads = 16

device = torch.device('cuda')
dtype = torch.float32

neurons = {
    'n0': LayerNormNeuron(shape=(n_tokens, n_embed), device=device, dtype=dtype)
}

synapses = {
    'attention': CausalSelfAttentionSynapse(n_tokens, n_embed, n_heads, device=device, dtype=dtype),
    'memory': ImplicitHopfieldSynapse(n_embed, 4*n_embed, device=device, dtype=dtype)
}

connections = {
    'attention': ['n0'],
    'memory': ['n0']
}

sensors = {
    'n0': nn.Embedding(n_vocab, n_embed, device=device, dtype=dtype)
}

outputs = {
    'n0': nn.Linear(n_embed, n_vocab, device=device, dtype=dtype)
}

sensors['n0'].weight = outputs['n0'].weight

model = HAM(
    neurons,
    synapses,
    connections,
    sensors,
    outputs
)

print(f'#params = {sum(p.numel() for p in model.parameters())/1e9:3.4f}B', flush=True)

def get_batch(x, batch_size, n_tokens, device, dtype):
    
    js = np.random.randint(0, x.shape[0] - n_tokens - 1, size=batch_size)
    xs = [torch.tensor(x[j:j+n_tokens].astype(np.int64), dtype=torch.long) for j in js]
    ys = [torch.tensor(x[j+n_tokens:j+n_tokens+1].astype(np.int64), dtype=torch.long) for j in js]
    xs = torch.cat([v.unsqueeze(0) for v in xs], dim=0).to(device)
    ys = torch.cat([v.unsqueeze(0) for v in ys], dim=0).to(device)
    
    return xs, ys

optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

batch_size = 20
n_examples = 1_000_000
n_batches = n_examples // batch_size
checkpoint_interval = 50_000
checkpoint_interval_batches = checkpoint_interval // batch_size
depth = 1
alpha = 0.1

checkpoint_dir = 'checkpoints'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
    print(f'created checkpoint directory: {checkpoint_dir}', flush=True)

for i in range(n_batches):
    
    optim.zero_grad()
    
    xs, ys = get_batch(data, batch_size, n_tokens, device, dtype)
    states = model.input({ 'n0': xs })
    for d in range(depth):
        activations = model.activations(states)
        updates = model.updates(states, activations)
        states = model.step(states, updates, alpha)
    
    logits = model.output(states)['n0'][:,-1,:]
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ys.view(-1), ignore_index=-1)
    loss.backward()
    optim.step()
    
    print(f'i = {i:08d}/{n_batches:08d}, loss = {loss.item():2.6f}', end='\r', flush=True)

    if (i+1) % checkpoint_interval_batches == 0:
        print('')
        savepath = os.path.join(checkpoint_dir, f'ckpt_{i+1:08d}.pt')
        torch.save({
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'batch_idx': i
        }, savepath)
        print(f'saved checkpoint: {savepath}', flush=True)