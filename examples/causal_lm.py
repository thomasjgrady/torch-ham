from dataclasses import dataclass, field
from torch import Tensor
from torch_ham import HAM
from torch_ham.neurons import SoftmaxNeuron, SigmoidNeuron
from torch_ham.synapses import AttentionSynapse, DenseSynapse, HopfieldSynapse, HopfieldWeight
from torch_ham.optimization import deq_fixed_point
from typing import *

import os
import math
import numpy as np
import torch

@dataclass
class ModelConfig:
    vocab_size: int = 50_257 # Model vocabulary size
    ctx_width: int = 128     # Maximum input sequence length
    n_embed: int = 128       # Embedding dimension
    n_heads: int = 16        # Number of attention heads
    n_hidden: int = 1024     # Hidden size in Hopfield synapse
    beta_out: float = 1.0    # Inverse temperature for outputs
    beta_mem: float = 10.0   # Inverse temperature for Hopfield memory

    device: torch.device = torch.device('cpu') # Device for training
    dtype: torch.dtype = torch.float32         # Datatype for training

def create_model(config: ModelConfig) -> HAM:

    neurons = {
        'input': SoftmaxNeuron(shape=(config.ctx_width, config.vocab_size)),
        'output': SoftmaxNeuron(shape=(config.ctx_width, config.vocab_size), beta=config.beta_out),
        'query': SigmoidNeuron(shape=(config.ctx_width, config.n_embed)),
        'key': SigmoidNeuron(shape=(config.ctx_width, config.n_embed))
    }

    synapses = {
        'encode': DenseSynapse(config.vocab_size, config.n_embed, device=config.device, dtype=config.dtype),
        'decode': DenseSynapse(config.vocab_size, config.n_embed, device=config.device, dtype=config.dtype),
        'hop_query': HopfieldSynapse(HopfieldWeight(config.n_embed, config.n_hidden, device=config.device, dtype=config.dtype), beta=config.beta_mem),
        'hop_key': HopfieldSynapse(HopfieldWeight(config.n_embed, config.n_hidden, device=config.device, dtype=config.dtype), beta=config.beta_mem),
        'attn': AttentionSynapse(
            n_embed_q=config.n_embed,
            n_embed_k=config.n_embed,
            n_embed=config.n_embed,
            n_heads=config.n_heads,
            ctx_width=config.ctx_width,
            device=config.device,
            dtype=config.dtype
        )
    }

    # Weight tying
    synapses['encode'].W = synapses['decode'].W

    connections = {
        'encode': ['input', 'key'],
        'decode': ['output', 'query'],
        'hop_query': ['query'],
        'hop_key': ['key'],
        'attn': ['query', 'key']
    }

    return HAM(neurons, synapses, connections)

@dataclass
class TrainConfig:

    train_path: str = os.path.expanduser('~/data/wikitext-103/train.npy') # Path to training tokens (uint16 numpy format)
    valid_path: str = os.path.expanduser('~/data/wikitext-103/valid.npy') # Path to validation tokens (uint16 numpy format)

    batch_size: int = 1                 # Training batch size
    n_examples: int = 1_000_000         # Number of training examples
    n_batches: int  = field(init=False) # Number of training batches

    min_lr: float     = 1e-5 # Minimum learning rate
    max_lr: float     = 1e-4 # Maximum learning rate
    warmup: int       = 10_000 # Learning rate warmup in number of examples
    warmup_batch: int = field(init=False) # Learning rate warmup in number of batches
    decay: float      = 1e-4 # Weight decay

    deq_max_iter: int    = 10  # Maximum number of deep equilibrium iterations during training (inner loop)
    deq_step_size: float = 1.0 # Step size for energy descent in deq training iter
    deq_tol: float       = 1e-3 # Convergence threshold for deq

    def __post_init__(self) -> None:
        self.n_batches = self.n_examples // self.batch_size
        self.warmup_batch = self.warmup // self.batch_size

def get_lr(batch_idx: int, config: TrainConfig) -> float:
    if batch_idx < config.warmup_batch:
        return math.sin(batch_idx/config.warmup_batch)*config.max_lr
    else:
        frac = math.cos((batch_idx-config.warmup_batch)/(config.n_batches-config.warmup_batch))
        return frac*config.max_lr + (1-frac)*config.min_lr

def get_train_batch(train_data: np.ndarray, train_config: TrainConfig, model_config: ModelConfig) -> Tuple[Tensor, Tensor]:
    starts = np.random.randint(low=0, high=len(train_data)-model_config.ctx_width-1, size=train_config.batch_size)
    examples = torch.cat([torch.tensor(train_data[s:s+model_config.ctx_width+1].astype(np.int64)).unsqueeze(0) for s in starts], dim=0) \
        .to(device=model_config.device, dtype=torch.long)
    return examples[:,:-1], examples[:,-1]

if __name__ == '__main__':

    from collections import defaultdict
    from simple_parsing import ArgumentParser
    from tqdm.auto import tqdm

    import torch.nn.functional as F

    parser = ArgumentParser()
    parser.add_arguments(ModelConfig, dest='model')
    parser.add_arguments(TrainConfig, dest='train')
    args = parser.parse_args()

    # Memmap training and valid data
    train_data = np.memmap(args.train.train_path)
    #valid_data = np.memmap(args.train.valid_path)

    # Create model
    model = create_model(args.model)

    # Setup optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.train.min_lr, weight_decay=args.train.decay)

    pbar = tqdm(range(args.train.n_batches))
    for i in pbar:

        lr = get_lr(i, args.train)
        for g in optim.param_groups:
            g['lr'] = lr

        x, y = get_train_batch(train_data, args.train, args.model)
        states = model.init_states(
            batch_size=x.shape[0],
            exclude={'input'},
            device=args.model.device,
            dtype=args.model.dtype,
            requires_grad=True
        )
        states['input'] = F.one_hot(x, num_classes=args.model.vocab_size) \
            .to(args.model.dtype) \
            .requires_grad_() + 0.001
        activations = model.activations(states)

        states, activations = deq_fixed_point(
            model,
            states,
            activations,
            max_iter=args.train.deq_max_iter,
            alpha=defaultdict(lambda: args.train.deq_step_size),
            pin={'input'},
            tol=args.train.deq_tol
        )

        logprobs = torch.log(activations['output'][:,-1,:].flatten(start_dim=1))

        loss = F.nll_loss(logprobs, y.view(-1), ignore_index=-1)
        optim.zero_grad()
        loss.backward()
        optim.step()

        pbar.set_description(f'loss = {loss.item()}')