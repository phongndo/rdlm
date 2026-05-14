"""
tiny_block.py — The 2-layer recursive building block for TRM.

This is the "tiny network" that gets applied repeatedly.
It's a minimal 2-layer Transformer with:
  Layer 1: Multi-Head Self-Attention (with RoPE)
  Layer 2: SwiGLU Feed-Forward Network

Both layers have Pre-Norm (RMSNorm) and residual connections.
No biases anywhere (following modern practice).
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange


def exists(v):
    return v is not None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — simpler than LayerNorm, no bias."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, dim)
        rms = x.pow(2).mean(-1, keepdim=True).clamp(min=1e-6).sqrt()
        return x / rms * self.scale


def precompute_rope_frequencies(dim: int, max_len: int, theta: float = 10000.0) -> Tensor:
    """Precompute Rotary Position Embedding frequencies.

    Args:
        dim: Head dimension (must be even)
        max_len: Maximum sequence length to precompute for
        theta: RoPE base frequency
    Returns:
        cos, sin: Both of shape (max_len, dim)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_len).float()
    angles = torch.outer(positions, freqs)  # (max_len, dim/2)
    angles = torch.cat([angles, angles], dim=-1)  # (max_len, dim)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings to queries or keys.

    Args:
        x: (batch, num_heads, seq_len, head_dim)
        cos, sin: (seq_len, head_dim)
    Returns:
        Rotated x of same shape
    """
    seq_len = x.shape[-2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
    return x * cos + x_rot * sin


class Attention(nn.Module):
    """Multi-head self-attention with RoPE, no bias."""

    def __init__(self, dim: int, num_heads: int = 4, head_dim: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.inner_dim = self.num_heads * self.head_dim

        # Combined QKV projection (no bias)
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

        # Precomputed RoPE (will be populated at first forward)
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, seq_len, _ = x.shape

        # Lazily compute RoPE for the max sequence length we've seen
        if self.rope_cos is None or seq_len > self.rope_cos.shape[0]:
            cos, sin = precompute_rope_frequencies(self.head_dim, seq_len)
            self.register_buffer("rope_cos", cos.to(x.device), persistent=False)
            self.register_buffer("rope_sin", sin.to(x.device), persistent=False)

        # QKV projection
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads)

        # Apply RoPE
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Mask out padding positions (mask: (batch, seq_len), 1 = valid, 0 = pad)
        if mask is not None:
            attn_mask = mask[:, None, None, :]  # (batch, 1, 1, seq_len)
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: Swish-gated linear unit.

    SwiGLU(x) = (xW₁ ⊙ σ(xW₂)) · W₃
    where σ is SiLU (Swish), and ⊙ is element-wise multiplication.
    """

    def __init__(self, dim: int, hidden_multiple: int = 4):
        super().__init__()
        hidden_dim = dim * hidden_multiple
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU: (xW₁) * SiLU(xW₂) projected back
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = x1 * F.silu(x2)
        return self.w3(hidden)


class TinyBlock(nn.Module):
    """The 2-layer recursive block for TRM.

    Architecture:
      x → RMSNorm → Attention → ✕ α_attn → + → RMSNorm → SwiGLU → ✕ α_ff → + → out

    Each sublayer has a **learnable residual scale** (like ReZero):
    - Initialized to 0 so the block starts as identity (prevents blow-up)
    - Gradients flow through because the scale is a learnable parameter
    - The model learns how much each sublayer should contribute

    Reference: ReZero is All You Need (https://arxiv.org/abs/2003.04887)
    """

    def __init__(self, dim: int, num_heads: int = 4, head_dim: int | None = None, ff_mult: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, head_dim=head_dim)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, hidden_multiple=ff_mult)

        # ── Learnable residual scales ───────────────────────────────
        # These control how much each sublayer contributes to the residual.
        # Initialized to a small positive value (1e-3) so that gradients
        # can flow through to the sublayer parameters from step 1.
        # The model can then learn to increase or decrease these as needed.
        self.alpha_attn = nn.Parameter(torch.tensor(1e-3))
        self.alpha_ff = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Apply the 2-layer block with residual connections.

        Args:
            x: (batch, seq_len, dim) — input representation
            mask: (batch, seq_len) or None — 1 = valid, 0 = padding
        Returns:
            Tensor of same shape as input
        """
        # Layer 1: Self-attention with pre-norm + learnable residual scale
        x = x + self.alpha_attn * self.attn(self.norm1(x), mask=mask)

        # Layer 2: SwiGLU feed-forward with pre-norm + learnable residual scale
        x = x + self.alpha_ff * self.ff(self.norm2(x))

        return x


if __name__ == "__main__":
    # Quick sanity check
    batch, seq_len, dim = 2, 16, 128
    block = TinyBlock(dim=dim, num_heads=4)
    x = torch.randn(batch, seq_len, dim)
    out = block(x)
    print(f"TinyBlock output shape: {out.shape}")  # Expected: (2, 16, 128)
    assert out.shape == x.shape, "Shape should be preserved"
    print("✅ TinyBlock works!")
