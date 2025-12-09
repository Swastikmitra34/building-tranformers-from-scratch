import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Single-head causal self-attention.

    Input: x of shape (B, T, C)
    Output: same shape (B, T, C)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q(x)   # (B, T, C)
        k = self.k(x)   # (B, T, C)
        v = self.v(x)   # (B, T, C)

        # Attention scores: (B, T, T)
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)

        # Causal mask: allow only j <= i
        mask = torch.tril(torch.ones(T, T, device=x.device))
        wei = wei.masked_fill(mask == 0, float("-inf"))

        # Softmax over last dimension (keys)
        attn = F.softmax(wei, dim=-1)   # (B, T, T)

        # Weighted sum of values
        out = attn @ v                  # (B, T, C)
        return out


class Block(nn.Module):
    """
    One Transformer block:
    - LayerNorm
    - Self-Attention
    - Residual
    - LayerNorm
    - Feed-Forward
    - Residual
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.sa = SelfAttention(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """
    Minimal GPT-style Transformer language model.

    - Character embeddings
    - Learned positional embeddings
    - Several Transformer blocks
    - Final linear head projecting to vocab
    """

    def __init__(self, vocab_size: int, embed_dim: int, block_size: int, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx):
        """
        idx: (B, T) of token indices
        returns logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.block_size}")

        # Token + positional embeddings
        token_emb = self.token_emb(idx)  # (B, T, C)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)[None, :, :]  # (1, T, C)
        x = token_emb + pos_emb                  # (B, T, C)

        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)

        # Project to logits
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """
        Autoregressive generation.

        idx: (B, T_start)
        Returns: (B, T_start + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # crop to last block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # get logits for last position
            logits = self(idx_cond)           # (B, T, vocab_size)
            logits = logits[:, -1, :]         # (B, vocab_size)

            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat((idx, next_id), dim=1)             # (B, T+1)
        return idx
