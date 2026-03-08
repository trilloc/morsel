#!/usr/bin/env python3
"""
morsel — a really, really tiny LLM with no attention

The entire sequence mixing mechanism is one equation:

    h[t] = α · h[t-1] + (1 - α) · x[t]

Just a learned exponential moving average with per-channel decay rates.

Architecture:  Embed → [LayerNorm → EMA → MLP]×N → LM Head
Vocab:         byte-level (256) — no tokenizer needed

Usage:
    python morsel.py <textfile> "your prompt here"

Example:
    python morsel.py shakespeare.txt "ROMEO:"

Author:  @raskal Rajesh Bhaskar
License: MIT
"""

import sys, math, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")

# ── Config ────────────────────────────────────────────────────────────────────
V    = 256    # byte-level vocab — every byte is a token, no tokenizer
D    = 128    # model dimension
NL   = 10    # number of EMA + MLP blocks
ML   = 4      # MLP expansion factor
S    = 128    # context window length
BS   = 16      # mini-batch size
LR   = 1e-3   # Adam learning rate
STEPS = 5000  # training steps

# ── EMA: The Attention Replacement ───────────────────────────────────────────
# Where a Transformer does:  Attention(Q, K, V) = softmax(QKᵀ/√d)V   — O(n²)
# morsel does:                h[t] = α·h[t-1] + (1-α)·x[t]           — O(n)
class EMA(nn.Module):
    """Learned Exponential Moving Average — replaces self-attention.

    Each of the D channels gets its own decay rate α ∈ (0,1).
    α→1: long memory (slow decay)
    α→0: short memory (fast decay)
    """
    def __init__(self, d):
        super().__init__()
        # logit=1.0 → sigmoid(1)≈0.73, a reasonable starting decay
        self.alpha_logit = nn.Parameter(torch.ones(d))

    def forward(self, x):  # x: [B, T, D]
        alpha = torch.sigmoid(self.alpha_logit)
        B, T, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        out = []
        for t in range(T):
            h = alpha * h + (1 - alpha) * x[:, t]
            out.append(h)
        return torch.stack(out, dim=1)

# ── Block: LayerNorm → EMA → MLP ─────────────────────────────────────────────
# Same pre-norm residual structure as a Transformer block,
# but EMA replaces multi-head self-attention.
class Block(nn.Module):
    def __init__(self, d, mlp_mult):
        super().__init__()
        self.ln  = nn.LayerNorm(d)
        self.ema = EMA(d)
        self.w1  = nn.Linear(d, d * mlp_mult)  # up-projection
        self.w2  = nn.Linear(d * mlp_mult, d)   # down-projection

    def forward(self, x):
        xn = self.ln(x)            # normalize
        xr = xn + self.ema(xn)     # mix across time via EMA + residual
        return x + self.w2(F.gelu(self.w1(xr)))  # MLP + residual


# ── Model ─────────────────────────────────────────────────────────────────────
class Morsel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb    = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block(D, ML) for _ in range(NL)])
        self.head   = nn.Linear(D, V)

    def forward(self, x):  # x: [B, T] long tensor of byte values
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)  # [B, T, V] logits

# ── Generation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, prompt_bytes, n=200, temp=0.8):
    """Autoregressive byte-level generation."""
    ctx = list(prompt_bytes)
    for _ in range(n):
        inp = torch.tensor([ctx[-S:]], dtype=torch.long, device=device)
        logits = model(inp)[0, -1] / max(temp, 1e-3)
        tok = torch.multinomial(F.softmax(logits, -1), 1).item()
        ctx.append(tok)
    return bytes(ctx[len(prompt_bytes):])

# ── Main ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 3:
    print("  morsel — an LLM with no attention")
    print('  Usage: python morsel.py <textfile> "prompt"')
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Morsel().to(device)
nparams = sum(p.numel() for p in model.parameters())

# torch.compile for training; raw model for generation (avoids recompiles
# from changing sequence lengths during autoregressive decoding)
train_model = torch.compile(model) if hasattr(torch, "compile") else model
opt = torch.optim.Adam(model.parameters(), lr=LR)

fn = sys.argv[1]
prompt = " ".join(sys.argv[2:])

print(f"""
  ║  morsel — an LLM with no attention            
  ║  Params:    {nparams:>10,}                    
  ║  Layers:    {NL:>2} × (EMA + MLP)             
  ║  Dim:       {D:<4}  │  MLP: {D*ML:<4}         
  ║  Device:    {device:<10}                      
  ║  Core:      h[t] = α·h[t-1] + (1-α)·x[t]
  .. Compilation make some time.. Please wait while the model is being optimized for training..""")

# Load data as raw bytes
with open(fn, "rb") as f:
    data = torch.tensor(list(f.read()), dtype=torch.long, device=device)
print(f"\n  Data: {len(data):,} bytes from '{fn}'")
print(f"  Training: {STEPS} steps, batch={BS}, seq={S}\n")

# ── Training ──────────────────────────────────────────────────────────────────
t0 = time.time()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(data) - S - 1, (BS,), device=device)
    batch = torch.stack([data[i:i+S+1] for i in idx])

    logits = train_model(batch[:, :-1])
    loss = F.cross_entropy(logits.view(-1, V), batch[:, 1:].reshape(-1))

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0 or step == 1:
        dt = time.time() - t0
        tps = step * BS * S / dt
        print(f"  step {step:5d}/{STEPS} │ loss {loss.item():.4f} │ "
              f"bpc {loss.item()/math.log(2):.2f} │ {tps:.0f} tok/s │ {dt:.1f}s")

dt = time.time() - t0
print(f"\n  Training complete in {dt:.1f}s │ final loss {loss.item():.4f}")

# ── Generate ──────────────────────────────────────────────────────────────────
print(f"\n  Prompt: \"{prompt}\"")
print(f"  {'─'*50}")
out = generate(model, prompt.encode("utf-8"))
print(f"  {prompt}{out.decode('utf-8', errors='replace')}")
print(f"  {'─'*50}\n")
