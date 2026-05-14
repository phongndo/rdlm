"""
trm.py — Tiny Recursive Model for text.

Wraps a TinyBlock and applies it recursively:
  - Inner loop: refine the latent representation n times per block
  - Outer loop: T "deep supervision" blocks, each tries to predict the answer
  - Only the last outer step gets gradients (saves memory)

This is the text-adapted version of the TRM architecture from:
  "Less is More: Recursive Reasoning with Tiny Networks" (Samsung SAIL, 2025)
"""

import torch
from torch import nn, Tensor
from einops import rearrange


def exists(v):
    return v is not None


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + MLP projection.

    Maps a scalar timestep t ∈ [0, 1] to a dim-dimensional vector.
    Uses the standard diffusion-model approach:
      sin/cos of frequencies → SiLU → Linear → dim
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        half_dim = dim // 2
        freqs = torch.exp(-torch.arange(half_dim).float() * torch.log(torch.tensor(max_period)) / (half_dim - 1))
        self.register_buffer("freqs", freqs, persistent=False)

        self.net = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed timesteps.

        Args:
            t: (batch,) — timesteps in [0, 1]
        Returns:
            (batch, dim) — timestep embeddings
        """
        # Sinusoidal encoding
        t = t.unsqueeze(-1)  # (batch, 1)
        angles = t * self.freqs.unsqueeze(0)  # (batch, half_dim)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (batch, dim)
        return self.net(emb)


class TinyRecursiveModel(nn.Module):
    """Tiny Recursive Model for text sequences.

    The core idea: a single tiny 2-layer block is applied recursively,
    giving us the effective depth of a 30-layer network with only 2 layers of params.

    This supports both autoregressive and diffusion-based usage:
    - For autoregressive: just pass tokens, no timestep needed
    - For diffusion: pass tokens (can be masked) + timestep for noise-level conditioning

    Args:
        vocab_size: Size of the token vocabulary
        dim: Hidden dimension of the model
        max_seq_len: Maximum sequence length (for position embeddings)
        num_heads: Number of attention heads
        head_dim: Dimension per head (defaults to dim // num_heads)
        ff_mult: Feed-forward hidden multiplier
        num_latent_refinements: n — latent refinement steps per block (default 6)
        num_refinement_blocks: T — deep supervision blocks (default 3)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        max_seq_len: int = 512,
        num_heads: int = 4,
        head_dim: int | None = None,
        ff_mult: int = 4,
        num_latent_refinements: int = 6,
        num_refinement_blocks: int = 3,
    ):
        super().__init__()
        assert num_refinement_blocks >= 1, "Need at least 1 refinement block"
        assert num_latent_refinements >= 1, "Need at least 1 latent refinement"

        self.vocab_size = vocab_size
        self.dim = dim
        self.num_latent_refinements = num_latent_refinements
        self.num_refinement_blocks = num_refinement_blocks

        # ── Embeddings ────────────────────────────────────────────────
        # Token embedding: maps token IDs to dense vectors
        self.token_embed = nn.Embedding(vocab_size, dim)

        # Learnable initial states (like TRM's "output_init_embed" and "latent_init_embed")
        # These are the starting points for the recursive refinement.
        self.output_init = nn.Parameter(torch.randn(dim) * 0.02)
        self.latent_init = nn.Parameter(torch.randn(dim) * 0.02)

        # Position embedding (learned absolute, for simplicity)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        # ── Timestep embedding (for diffusion) ────────────────────────
        # Maps a continuous timestep t ∈ [0,1] to a dim-dimensional vector.
        # Uses sinusoidal embeddings + MLP (standard in diffusion models).
        self.time_embed = TimestepEmbedding(dim)

        # Learnable mask token embedding (for masked diffusion)
        self.mask_embed = nn.Parameter(torch.randn(dim) * 0.02)

        # ── The tiny recursive block ──────────────────────────────────
        # This single block is applied over and over.
        # We import TinyBlock here to avoid circular deps.
        from rdlm.tiny_block import TinyBlock

        self.network = TinyBlock(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            ff_mult=ff_mult,
        )

        # ── Output head ───────────────────────────────────────────────
        # Projects from hidden dim → vocabulary logits
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)

        # ── Halting prediction head ───────────────────────────────────
        # Predicts "is my current answer correct?" so we can halt early.
        self.to_halt = nn.Sequential(
            nn.Linear(dim, 1, bias=False),
        )

        # Tie the output head weight with the token embedding (weight tying)
        # This is standard practice — the same matrix embeds and predicts.
        self.to_logits.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        # Zero-init the halt head to start with uniform halting probability
        nn.init.zeros_(self.to_halt[0].weight)

    @property
    def device(self):
        return next(self.parameters()).device

    # ── Forward helpers ──────────────────────────────────────────────

    def _embed(self, tokens: Tensor, mask_token_id: int | None = None) -> Tensor:
        """Embed tokens + add position encodings.

        Handles masked positions: if mask_token_id is given, any token equal
        to mask_token_id gets the learnable mask embedding instead.

        Args:
            tokens: (batch, seq_len) — token IDs
            mask_token_id: ID of the mask token, or None for no masking
        Returns:
            (batch, seq_len, dim) — embedded sequence
        """
        batch, seq_len = tokens.shape

        # Token embeddings (replace mask tokens with mask_embed)
        if mask_token_id is not None:
            is_mask = (tokens == mask_token_id)
            token_embeds = self.token_embed(tokens)
            # Where the token is [MASK], use the learnable mask embedding
            token_embeds = torch.where(
                is_mask.unsqueeze(-1),
                self.mask_embed.unsqueeze(0).unsqueeze(0).expand_as(token_embeds),
                token_embeds,
            )
        else:
            token_embeds = self.token_embed(tokens)

        # Position embeddings
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        pos_embeds = self.pos_embed(positions)

        return token_embeds + pos_embeds

    def _get_initial_state(self, batch: int, seq_len: int) -> tuple[Tensor, Tensor]:
        """Create the initial output and latent representations.

        These are learnable parameters broadcast across the batch/sequence.

        Returns:
            outputs: (batch, seq_len, dim)
            latents: (batch, seq_len, dim)
        """
        outputs = self.output_init.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
        latents = self.latent_init.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
        return outputs, latents

    def _refine_one_block(self, inputs: Tensor, outputs: Tensor, latents: Tensor) -> tuple[Tensor, Tensor]:
        """One deep-supervision block: n latent refinements + 1 output refinement.

        Args:
            inputs: (batch, seq_len, dim) — embedded input tokens
            outputs: (batch, seq_len, dim) — current output representation
            latents: (batch, seq_len, dim) — current latent representation
        Returns:
            outputs, latents — both updated
        """
        # Inner loop: refine the latent n times
        for _ in range(self.num_latent_refinements):
            # The network takes the sum of (outputs + latents + inputs)
            # This is the TRM trick: the block sees a merged view of all signals
            latents = self.network(outputs + latents + inputs)

        # Then refine the output once from the (now improved) latent
        outputs = self.network(outputs + latents)

        return outputs, latents

    def _deep_refinement(self, inputs: Tensor, outputs: Tensor, latents: Tensor) -> tuple[Tensor, Tensor]:
        """Run all T deep-supervision blocks.

        Only the last block receives gradients (the rest use torch.no_grad).
        This is TRM's "deep supervision" — it saves memory while still
        getting the benefit of multiple refinement rounds.

        Args:
            inputs, outputs, latents: same as _refine_one_block
        Returns:
            final outputs, latents
        """
        for step in range(self.num_refinement_blocks):
            is_last = step == self.num_refinement_blocks - 1
            context = torch.no_grad if not is_last else nullcontext

            with context():
                outputs, latents = self._refine_one_block(inputs, outputs, latents)

        return outputs, latents

    # ── Forward pass (training) ──────────────────────────────────────

    def forward(
        self,
        tokens: Tensor,
        timestep: Tensor | None = None,
        labels: Tensor | None = None,
        mask: Tensor | None = None,
        mask_token_id: int | None = None,
    ) -> Tensor | dict:
        """Forward pass with optional diffusion timestep conditioning.

        Args:
            tokens: (batch, seq_len) — input token IDs (can be partially masked)
            timestep: (batch,) or None — diffusion timesteps t ∈ [0, 1]
            labels: (batch, seq_len) or None — target token IDs for loss
            mask: (batch, seq_len) or None — 1 = valid, 0 = padding
            mask_token_id: ID of [MASK] token, or None
        Returns:
            If labels given: dict with 'loss', 'logits', 'halt_probs'
            If no labels: logits tensor
        """
        batch, seq_len = tokens.shape

        # 1. Embed inputs (handles mask tokens if mask_token_id is given)
        inputs = self._embed(tokens, mask_token_id=mask_token_id)

        # 2. Add timestep conditioning (for diffusion)
        if timestep is not None:
            # Timestep embedding projects to (batch, dim), then expand to sequence
            t_embed = self.time_embed(timestep)  # (batch, dim)
            t_embed = t_embed.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, dim)
            inputs = inputs + t_embed

        # 3. Initialize states
        outputs, latents = self._get_initial_state(batch, seq_len)

        # 4. Deep refinement (the recursive core)
        outputs, latents = self._deep_refinement(inputs, outputs, latents)

        # 5. Predict logits from refined outputs
        logits = self.to_logits(outputs)  # (batch, seq_len, vocab_size)

        # 6. Compute halting probability (mean over sequence)
        halt_logits = self.to_halt(outputs.mean(dim=1)).squeeze(-1)  # (batch,)

        if labels is None:
            return {
                "loss": None,
                "token_loss": None,
                "halt_loss": None,
                "logits": logits,
                "halt_probs": halt_logits.sigmoid(),
            }

        # ── Loss computation ──────────────────────────────────────────
        # Token prediction loss (cross-entropy on all positions)
        loss = nn.functional.cross_entropy(
            rearrange(logits, "b n l -> b l n"),
            labels,
            reduction="none",
        )
        # Mask out padding positions
        if mask is not None:
            loss = loss * mask.float()
            n_tokens = mask.sum()
        else:
            n_tokens = loss.numel()
        token_loss = loss.sum() / n_tokens

        # Halting loss: should the model have halted?
        # A sample is "correct" if all its tokens match
        is_correct = (logits.argmax(dim=-1) == labels)  # (batch, seq_len)
        if mask is not None:
            is_correct = is_correct | (~mask.bool())
        all_correct = is_correct.all(dim=-1).float()  # (batch,)
        halt_loss = nn.functional.binary_cross_entropy_with_logits(
            halt_logits, all_correct
        )

        total_loss = token_loss + halt_loss

        return {
            "loss": total_loss,
            "token_loss": token_loss,
            "halt_loss": halt_loss,
            "logits": logits,
            "halt_probs": halt_logits.sigmoid(),
        }

    # ── Generation (inference) ───────────────────────────────────────

    @torch.no_grad()
    def generate_autoregressive(
        self,
        prompt: Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> Tensor:
        """Generate tokens autoregressively (left-to-right).

        This is a simple baseline. For diffusion-based generation,
        use the diffusion sampler instead.

        Args:
            prompt: (batch, prompt_len) — initial tokens
            max_new_tokens: how many new tokens to generate
            temperature: sampling temperature (0 = greedy)
        Returns:
            (batch, prompt_len + max_new_tokens) — full generated sequence
        """
        batch = prompt.shape[0]
        device = prompt.device
        generated = prompt.clone()

        for _ in range(max_new_tokens):
            # Forward pass on current sequence
            out = self.forward(generated, timestep=None)  # returns dict
            logits = out["logits"]  # (batch, seq_len, vocab)

            # Get logits for the last position
            next_logits = logits[:, -1, :]  # (batch, vocab)

            # Sample or greedy
            if temperature > 0:
                probs = nn.functional.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

        return generated


# Context manager for torch.no_grad (used in _deep_refinement)
from contextlib import nullcontext


if __name__ == "__main__":
    # Quick sanity check
    vocab_size = 100
    dim = 128
    model = TinyRecursiveModel(
        vocab_size=vocab_size,
        dim=dim,
        num_latent_refinements=6,
        num_refinement_blocks=3,
    )

    batch, seq_len = 2, 16
    tokens = torch.randint(0, vocab_size, (batch, seq_len))
    labels = torch.randint(0, vocab_size, (batch, seq_len))

    out = model(tokens, labels=labels)
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"Token loss: {out['token_loss'].item():.4f}")
    print(f"Halt loss: {out['halt_loss'].item():.4f}")
    print(f"Logits shape: {out['logits'].shape}")
    print("✅ TinyRecursiveModel works!")

    # Quick generation test
    prompt = torch.randint(0, vocab_size, (1, 4))
    generated = model.generate_autoregressive(prompt, max_new_tokens=8)
    print(f"Prompt: {prompt.shape} → Generated: {generated.shape}")
