"""
morsel — a really, really tiny LLM with no attention

This is the tranformer version of morsel.

Usage:
    python morseltransformer.py <textfile> "prompt"

Example:
    python morseltransformer.py tinyshakespeare.txt "ROMEO:"

Author:  @raskal Rajesh Bhaskar
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

torch.set_float32_matmul_precision("high")

if len(sys.argv) < 3:
    print("  morseltransformer — tiny GPT baseline")
    print('  Usage: python morseltransformer.py <textfile> "prompt"')
    sys.exit(1)

data_file = sys.argv[1]
prompt = " ".join(sys.argv[2:])

# hyperparameters
batch_size = 64
block_size = 128
max_iters = 25000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_embd = 192
n_head = 8
n_layer = 3
dropout = 0.1

# dataset
with open(data_file, 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# attention head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))

        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next), dim=1)
        return idx

model = GPT().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, "M parameters")

# Compile only the training graph. Generation has changing sequence lengths,
# which can trigger recompiles and reduce/erase speed gains.
train_model = model
compile_enabled = False
if hasattr(torch, 'compile'):
    try:
        train_model = torch.compile(model)
        compile_enabled = True
    except Exception as e:
        print(f"torch.compile unavailable, using eager mode: {e}")

print(f"torch.compile: {'enabled' if compile_enabled else 'disabled'}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    xb,yb = get_batch('train')
    logits, loss = train_model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        print(iter, loss.item())

unknown_chars = sorted(set(ch for ch in prompt if ch not in stoi))
if unknown_chars:
    print(f"Prompt contains chars not in training text vocabulary: {unknown_chars}")
    sys.exit(1)

prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
generated_ids = model.generate(prompt_ids, 300)[0].tolist()
print(prompt + decode(generated_ids[len(prompt_ids[0]):]))