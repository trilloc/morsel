# morsel — an LLM with no attention

The entire sequence mixing mechanism is one equation:

```
h[t] = α · h[t-1] + (1 - α) · x[t]
```

No queries. No keys. No values. No softmax. No O(n²) memory.

Just a learned exponential moving average. ~150 lines of PyTorch. Single file. Train on a GPU in 5 minutes.

**morsel is the simplest possible language model in the SSM family** — no selection mechanism, no structured init, no complex dynamics. Just one learned decay rate per channel. It exists to answer the question: *what's the minimum that still learns language?*

## Quick start

```bash
pip install torch
python morsel.py shakespeare.txt "ROMEO:"
```

It reads the file, trains for 5000 steps, and generates text. That's it.

## What it does

morsel replaces self-attention with a **learned EMA scan**. Each of the `D` channels has its own decay rate `α ∈ (0,1)` that the model discovers from data:

- **α → 1**: channel remembers long (tracks context like "who is speaking")
- **α → 0**: channel forgets fast (tracks local patterns like word boundaries)

The model learns a spectrum of timescales on its own — no architecture search, no hyperparameter tuning of the decay rates.

## Architecture

```
Byte Embedding (256 → D)
       │
       ▼
┌──────────────────────┐
│  LayerNorm           │ ×N layers
│  EMA scan            │  ← this replaces attention
│  MLP (GELU)          │
│  + residual          │
└──────────────────────┘
       │
       ▼
  Linear (D → 256)
```

The vocab is byte-level — every byte is a token, no tokenizer needed.

## Sample output

Trained on Shakespeare (~1.1MB), ~1.4M parameters, RTX 4070, ~5 minutes:

```
ROMEO:
Twice to goes, we say you.
MISTRESS OVERDONE:
I will not to go drankness born.
HORTENSIO:
Shis wear?
SICINIUS:
I will not aim out. We stay;
To wish me, who shame food the holy behim,
And shall for
```

Character names, dialogue format, verse rhythm, Elizabethan vocabulary — from an EMA.

## Config

Edit the top of `morsel.py`:

```python
V    = 256    # byte-level vocab
D    = 128    # model dimension
NL   = 10    # layers
ML   = 4      # MLP expansion
S    = 128    # context length
BS   = 8      # batch size
LR   = 1e-3   # learning rate
STEPS = 5000  # training steps
```

## "Isn't this just RWKV / Mamba / S4?"

Yes, morsel belongs to the same family. Here's where it sits:

| Model | Sequence mixing | Key difference from morsel |
|---|---|---|
| **S4** | Structured state space (HiPPO init, complex poles) | Structured initialization, complex-valued states |
| **RWKV** | Token shift + channel mixing with learned decay | Time-mix and channel-mix as separate ops |
| **Mamba** | Selective SSM (input-dependent A, B, C) | Decay rate changes at every token based on content |
| **morsel** | Fixed learned EMA per channel | Nothing. This is the stripped-down base case. |

morsel is what you get when you remove every trick from the SSM family and ask: *what's the absolute minimum that still learns language?* One scalar decay rate per channel, constant across all inputs, no structured init, no selection mechanism, no complex dynamics.

The answer is: it still learns character names, dialogue structure, and verse rhythm at 5M params. That's the interesting part — not that it's novel, but that it's *sufficient*.

If you want to see what each addition buys you, morsel is your starting point. Add input-dependent α → you get Mamba's selection. Add complex poles → you get S4's oscillatory modes. Add token shift → you approach RWKV. Each addition is a few lines of code on top of `morsel.py`.

## What's next

morsel is the base case. We've been building on top of it:

- **Multi-stream architecture** — multiple parallel streams with different timescale characteristics, which solves the coherence collapse that a single EMA stream hits at longer contexts.
- **A theoretically grounded training objective** — not just cross-entropy.

Running on a single RTX 4070, producing coherent multi-paragraph text at 70M+ parameters. No attention.

## Requirements

```
pip install torch
```

That's all.

## License

MIT

## Author

Rajesh Bhaskar
