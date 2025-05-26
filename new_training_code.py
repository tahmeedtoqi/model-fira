# Install necessary modules (if not already installed)
!pip install safetensors
!pip install mpi4py
!pip install torchgpipe

import os
import time
import math
import csv
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import nullcontext
from tokenizers import Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import AdamW
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time
import safetensors.torch as st
import random

# Set random seeds for reproducibility
# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Default configuration values
new_dir = '/kaggle/input/input0'
out_dir = '/kaggle/working/'
eval_interval = 10
log_interval = 1
eval_iters = 10
eval_only = False
always_save_checkpoint = True
init_from = 'resume'  # options: 'scratch', 'resume', or 'gpt2'
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'FIRA'
dataset = 'fineweb-edu-10b/edu_fineweb10B'

batch_size = 16
block_size = 512  # Sequence length
n_layer = 12
n_head = 12
n_embd = 768
d_ff = 4 * n_embd
dropout = 0.1
num_experts = 4
learning_rate = 6e-4
max_iters = 30
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 715   # Reduced warmup to allow decay
lr_decay_iters = 2000
min_lr = 5e-5
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_flag = False  

# Build a config dictionary for logging
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
if 'num_experts' not in config_keys:
    config_keys.append('num_experts')
if 'd_ff' not in config_keys:
    config_keys.append('d_ff')
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Load custom tokenizer from JSON
tokenizer_path = '/kaggle/input/tokenz4/tokenizer.json'
tokenizer = Tokenizer.from_file(tokenizer_path)

# Automatically get the vocabulary size
my_vocab_size = tokenizer.get_vocab_size()
print(f"Loaded tokenizer with vocab_size = {my_vocab_size}")

# Define encode and decode functions using the tokenizer's methods
def encode(text):
    return tokenizer.encode(text).ids

def decode(ids):
    text = tokenizer.decode(ids)
    text = text.replace("Ġ", " ")
    special_tokens = ["[UNK]", "<s>", "</s>", "<|prompter|>", "<|assistant|>", "[PAD]", "<|endoftext|>"]
    for token in special_tokens:
        text = text.replace(token, "")
    return " ".join(text.split()).strip()

def compute_total_parameters(n_layer, n_head, n_embd, d_ff, num_experts):
    embedding_params = n_embd * my_vocab_size
    qkv_proj = 3 * n_embd * n_embd
    out_proj = n_embd * n_embd
    attn_params_per_layer = qkv_proj + out_proj
    expert_params = num_experts * (n_embd * d_ff + d_ff * n_embd)
    gate_params = n_embd * num_experts
    layernorm_params = 2 * 2 * n_embd
    total_per_layer = attn_params_per_layer + expert_params + gate_params + layernorm_params
    transformer_params = total_per_layer * n_layer
    final_ln_params = 2 * n_embd
    lm_head_params = n_embd * 50257
    total_params = embedding_params + transformer_params + final_ln_params + lm_head_params
    return total_params

total_params = compute_total_parameters(n_layer, n_head, n_embd, d_ff, num_experts)
print(f"{total_params / 1e6:.2f} Million parameters")

# -----------------------------------------------------------------------------
# Data loader class with state save/restore
data_dir = os.path.join('/kaggle/input/', dataset)

class DataLoaderLite:
    def __init__(self, B, T, split, data_dir):
        self.B = B
        self.T = T
        self.split = split
        self.data_dir = data_dir
        self.shards = self.get_shards()
        if len(self.shards) == 0:
            raise ValueError(f"No shards found for split '{split}' in {data_dir}")
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def get_shards(self):
        all_files = os.listdir(self.data_dir)
        if self.split == 'train':
            shards = [f for f in all_files if f.startswith("edufineweb_train") and f.endswith(".npy")]
        else:
            shards = [f for f in all_files if f.startswith("edufineweb_val") and f.endswith(".npy")]
        return sorted([os.path.join(self.data_dir, s) for s in shards])

    def load_tokens(self, shard_path):
        return np.memmap(shard_path, dtype=np.uint16, mode='r')

    def next_batch(self):
        B, T = self.B, self.T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        buf = self.tokens[self.current_position : self.current_position + B * T + 1].astype(np.int64)
        x = torch.from_numpy(buf[:-1].reshape(B, T)).to('cuda:0')
        y = torch.from_numpy(buf[1:].reshape(B, T)).to('cuda:0')
        x = torch.clamp(x, 0, my_vocab_size - 1)
        y = torch.clamp(y, 0, my_vocab_size - 1)
        self.current_position += B * T
        return x, y

    def get_state(self):
        return {'current_shard': self.current_shard, 'current_position': self.current_position}

    def set_state(self, state):
        self.current_shard = state['current_shard']
        self.current_position = state['current_position']
        self.tokens = self.load_tokens(self.shards[self.current_shard])

train_loader = DataLoaderLite(B=batch_size, T=block_size, split='train', data_dir=data_dir)
val_loader = DataLoaderLite(B=batch_size, T=block_size, split='val', data_dir=data_dir)

def get_batch(split):
    loader = train_loader if split == 'train' else val_loader
    return loader.next_batch()

def convert_pipeline_state_dict(state_dict, model):
    model_state = model.state_dict()
    pipe_state = state_dict
    model_shape_to_keys = {}
    for key, param in model_state.items():
        shape = tuple(param.shape)
        if shape not in model_shape_to_keys:
            model_shape_to_keys[shape] = []
        model_shape_to_keys[shape].append(key)
    pipe_shape_to_keys = {}
    for key, param in pipe_state.items():
        shape = tuple(param.shape)
        if shape not in pipe_shape_to_keys:
            pipe_shape_to_keys[shape] = []
        pipe_shape_to_keys[shape].append(key)
    new_state_dict = {}
    for shape, model_keys in model_shape_to_keys.items():
        if shape in pipe_shape_to_keys:
            pipe_keys = pipe_shape_to_keys[shape]
            if len(model_keys) != len(pipe_keys):
                raise ValueError(f"Mismatch in number of parameters with shape {shape}: model has {len(model_keys)}, pipeline has {len(pipe_keys)}")
            for model_key, pipe_key in zip(model_keys, pipe_keys):
                new_state_dict[model_key] = pipe_state[pipe_key]
        else:
            raise ValueError(f"No parameters with shape {shape} in pipeline state_dict")
    return new_state_dict

model_args = dict(
    num_layers=n_layer,
    n_head=n_head,
    d_model=n_embd,
    d_ff=d_ff,
    num_experts=num_experts,
    max_seq_len=block_size,
    dropout=dropout,
    vocab_size=my_vocab_size
)

# -----------------------------------------------------------------------------
# Model Initialization / Checkpoint Restoration
# (Assuming FIRA is defined elsewhere)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model = FIRA(**model_args)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'checkpoint.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model_args = checkpoint['model_args']
    if checkpoint_model_args['vocab_size'] != my_vocab_size:
        raise ValueError("Checkpoint vocab_size mismatch")
    model = FIRA(**checkpoint_model_args)
    state_dict = st.load_file(os.path.join(out_dir, 'model.safetensors'), device='cpu')
    print("Original keys:", list(state_dict.keys())[:5])
    state_dict = convert_pipeline_state_dict(state_dict, model)
    print("Converted keys:", list(state_dict.keys())[:5])
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    # Restore last losses if available
    last_train_loss = checkpoint.get('last_train_loss', None)
    last_val_loss = checkpoint.get('last_val_loss', None)
    # Restore DataLoader states
    if 'data_loader_state' in checkpoint:
        train_loader.set_state(checkpoint['data_loader_state']['train'])
        val_loader.set_state(checkpoint['data_loader_state']['val'])
    # Restore RNG states
    random.setstate(checkpoint['rng_state_python'])
    np.random.set_state(checkpoint['rng_state_numpy'])
    torch.set_rng_state(checkpoint['rng_state_torch'])
    if torch.cuda.is_available() and checkpoint['rng_state_cuda'] is not None:
        torch.cuda.set_rng_state_all(checkpoint['rng_state_cuda'])
    print(f"Resumed from step {iter_num}")
    if last_train_loss is not None and last_val_loss is not None:
        print(f"Previous losses — Train: {last_train_loss:.4f}, Val: {last_val_loss:.4f}")
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout, num_experts=num_experts)
    model = FIRA.from_pretrained(init_from, num_experts=num_experts, override_args=override_args)
    model_args = model.config.copy()

if init_from != 'resume':
    iter_num = 0
    best_val_loss = float('inf')
    last_train_loss = None
    last_val_loss = None

model.to('cpu')
# -----------------------------------------------------------------------------
# Build Pipeline-Parallel Model and Optimizer
class EmbeddingStage(nn.Module):
    def __init__(self, token_embedding, position_embedding, dropout, max_seq_len):
        super().__init__()
        self.token_embedding = token_embedding
        self.position_embedding = position_embedding
        self.dropout = dropout
        self.max_seq_len = max_seq_len

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.dropout(x)
        return x

def create_pipeline_model(model):
    num_layers = model.config['num_layers']
    embedding_stage = EmbeddingStage(
        model.token_embedding, model.position_embedding, model.dropout, model.config['max_seq_len']
    )
    stage0 = nn.Sequential(embedding_stage, *list(model.layers)[:num_layers // 2])
    stage1 = nn.Sequential(*list(model.layers)[num_layers // 2:], model.ln_f, model.lm_head)
    sequential_model = nn.Sequential(stage0, stage1)
    sample = torch.randint(0, my_vocab_size, (batch_size, block_size)).to('cuda:0')
    balance = balance_by_time(2, sequential_model, sample)
    print("Computed balance:", balance)
    pipe_model = GPipe(sequential_model, balance=balance, devices=['cuda:0', 'cuda:1'], chunks=6)
    return pipe_model

pipe_model = create_pipeline_model(model)
pipe_model.train()
optimizer = model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(beta1, beta2),
    device_type=device
)

if compile_flag:
    try:
        pipe_model = torch.compile(pipe_model, mode='reduce-overhead')
        print("Model compiled successfully")
    except Exception as e:
        print(f"Compilation failed: {e}")
        compile_flag = False

"""Helper function for hellaswag"""
def get_most_likely_row(tokens, mask, logits):
    logits = logits.to(tokens.device)
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# Compute total batch tokens and gradient accumulation steps
total_batch_tokens = 524288  # desired total tokens per update (0.5M tokens)
ddp_world_size = 1 
ddp_rank = 0
assert total_batch_tokens % (batch_size * block_size * ddp_world_size) == 0, \
    "total_batch_tokens must be divisible by (batch_size * block_size * ddp_world_size)"
gradient_accumulation_steps = total_batch_tokens // (batch_size * block_size * ddp_world_size)
print(f"Total desired batch tokens: {total_batch_tokens}, gradient accumulation steps: {gradient_accumulation_steps}")

# -----------------------------------------------------------------------------
# Generation Function for Inference
def generate_text(prompt, num_return_sequences=4, max_length=32, temperature=1.0, top_k=50):
    pipe_model.eval()
    with torch.no_grad():
        tokens = torch.tensor([encode(prompt)], dtype=torch.long).to("cuda:0")
        tokens = tokens.repeat(num_return_sequences, 1)
        sample_rng = torch.Generator(device="cuda:0")
        sample_rng.manual_seed(42)
        while tokens.size(1) < max_length:
            if tokens.size(1) > max_length:
                tokens = tokens[:, -max_length:]
            with torch.autocast("cuda", dtype=torch.float16 if dtype=="float16" else torch.bfloat16):
                logits = pipe_model(tokens)
            last_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                topk_probs, topk_indices = torch.topk(F.softmax(last_logits, dim=-1), top_k, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
                next_token = torch.gather(topk_indices, -1, ix)
            else:
                probabilities = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1, generator=sample_rng)
            tokens = torch.cat([tokens, next_token.to(tokens.device)], dim=1)
        generated = decode(tokens[0, :max_length].tolist())
    pipe_model.train()
    return generated

# -----------------------------------------------------------------------------
# Utility Functions
@torch.no_grad()
def estimate_loss():
    pipe_model.eval()
    out = {}
    losses = {'train': [], 'val': []}
    for split in ['train', 'val']:
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast('cuda', dtype=torch.float16 if dtype=='float16' else torch.bfloat16):
                logits = pipe_model(X)
                Y = Y.to(logits.device)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[split].append(loss.item())
        out[split] = sum(losses[split]) / len(losses[split])
    pipe_model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

log_file = os.path.join(out_dir, 'training_log.csv')
def log_to_csv(iter_num, train_loss=None, val_loss=None, lr_val=None, mfu=None, hella_acc=None):
    file_exists = os.path.isfile(log_file)
    fieldnames = ['iter', 'train_loss', 'val_loss', 'lr', 'mfu']
    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'iter': iter_num,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': lr_val,
            'mfu': mfu
        })

# -----------------------------------------------------------------------------
# Training Loop
total_tokens = 0  # Initialize total tokens trained

# Set iter_num and best_val_loss based on checkpoint (if not resuming, they are set above)
running_mfu = -1.0  # Placeholder for MFU calculation

while iter_num < max_iters:
    current_lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        log_to_csv(iter_num, train_loss=losses['train'], val_loss=losses['val'], lr_val=current_lr, mfu=running_mfu*100)
        # Generate inference text after evaluation.
        prompt = "Hello my name is Fira, I am an ai assistent"
        generated_text = generate_text(prompt, max_length=32, temperature=1.0, top_k=50)
        print("Generated text:", generated_text)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            state_dict = pipe_model.state_dict()
            st.save_file(state_dict, os.path.join(out_dir, 'model.safetensors'))
            checkpoint = {
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'last_train_loss': losses['train'],
                'last_val_loss': losses['val'],
                'config': config,
                'data_loader_state': {
                    'train': train_loader.get_state(),
                    'val': val_loader.get_state()
                },
                'rng_state_python': random.getstate(),
                'rng_state_numpy': np.random.get_state(),
                'rng_state_torch': torch.get_rng_state(),
                'rng_state_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint.pt'))

    # Training step with gradient accumulation
    pipe_model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    t0 = time.time()
    for micro_step in range(gradient_accumulation_steps):
        X, Y = get_batch('train')
        with torch.autocast('cuda', dtype=torch.float16 if dtype == 'float16' else torch.bfloat16):
            logits = pipe_model(X)
            Y = Y.to(logits.device)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1)) / gradient_accumulation_steps
        loss.backward()
        loss_accum += loss.item()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(pipe_model.parameters(), grad_clip)
    optimizer.step()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = gradient_accumulation_steps * batch_size * block_size
    tokens_per_sec = tokens_processed / dt
    total_tokens += tokens_processed
    if iter_num % log_interval == 0:
        print(f"step {iter_num}: loss {loss_accum:.4f}, iteration time {dt:.4f} s, tokens/sec {tokens_per_sec:.2f}")
        log_to_csv(iter_num, train_loss=loss_accum, lr_val=current_lr, mfu=running_mfu*100)
    iter_num += 1

print(f"Total tokens trained: {total_tokens / 1e6:.2f} million")

df = pd.read_csv(log_file)
for col in ['train_loss', 'val_loss']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
plt.figure(figsize=(10, 5))
if 'train_loss' in df.columns and not df['train_loss'].isna().all():
    plt.plot(df['iter'], df['train_loss'], label='Train Loss')
if 'val_loss' in df.columns and not df['val_loss'].isna().all():
    plt.plot(df['iter'], df['val_loss'], label='Val Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')
plt.savefig(os.path.join(out_dir, 'training_progress.png'))
plt.close()
print(f"Training progress plot saved to {os.path.join(out_dir, 'training_progress.png')}")
