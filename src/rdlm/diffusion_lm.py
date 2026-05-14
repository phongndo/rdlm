"""
diffusion_lm.py — Recursive Diffusion Language Model.

Wraps a TinyRecursiveModel as a masked diffusion language model (MDLM).

Training:
  1. Take a clean sequence x₀
  2. Sample timestep t ~ Uniform(0, 1)
  3. Mask each token independently with probability t → x_t
  4. Model predicts clean tokens from x_t
  5. Loss is cross-entropy only on masked positions

Inference (generation):
  1. Start with a fully masked sequence
  2. For each diffusion step:
     a. Model predicts clean tokens for all positions
     b. Compute confidence (softmax probability) for each prediction
     c. Keep the top-k most confident predictions, re-mask the rest
  3. Repeat until all tokens are revealed

Reference:
  - Simple and Effective Masked Diffusion Language Models (MDLM)
  - dLLM: Simple Diffusion Language Modeling (https://arxiv.org/abs/2602.22661)
"""

import math

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class NoiseSchedule:
    """Linear noise schedule for masked diffusion.

    At timestep t ∈ [0, 1], each token is masked independently with probability t.
    """

    def __init__(self, t_min: float = 0.0, t_max: float = 1.0):
        self.t_min = t_min
        self.t_max = t_max

    def sample_t(self, batch_size: int, device: str = "cpu") -> Tensor:
        """Sample timesteps uniformly in [t_min, t_max]."""
        t = torch.rand(batch_size, device=device)
        return t * (self.t_max - self.t_min) + self.t_min

    def mask_prob(self, t: Tensor) -> Tensor:
        """Probability of masking at timestep t."""
        return t  # linear schedule: p(mask) = t

    def get_num_transfer_tokens(
        self, mask_index: Tensor, steps: int, stochastic: bool = False
    ) -> Tensor:
        """Compute how many tokens to reveal at each step.

        Uses a cosine schedule: more tokens revealed in early steps,
        finer refinement in later steps.

        Args:
            mask_index: (batch, seq_len) — True = masked position
            steps: total number of diffusion steps
            stochastic: if True, add randomness to token counts
        Returns:
            (batch, steps) — number of tokens to reveal at each step
        """
        batch, _seq_len = mask_index.shape
        num_masks = mask_index.sum(dim=-1)  # (batch,)

        # Cosine schedule for unmasking
        step_ratios = torch.linspace(0, 1, steps + 1, device=mask_index.device)[1:]  # (steps,)
        schedule = 1 - (step_ratios * math.pi / 2).cos()  # (steps,)

        # Number of tokens to have unmasked at each step
        target_counts = (num_masks.unsqueeze(-1) * schedule.unsqueeze(0)).long()  # (batch, steps)

        # Clip to valid range (use torch.clamp with tensor bounds)
        zero = torch.zeros_like(num_masks.unsqueeze(-1))
        target_counts = torch.clamp(target_counts, zero, num_masks.unsqueeze(-1))

        # Tokens to reveal THIS step = cumulative[t] - cumulative[t-1]
        prev_counts = torch.cat([
            torch.zeros(batch, 1, device=mask_index.device, dtype=torch.long),
            target_counts[:, :-1],
        ], dim=-1)
        transfer = target_counts - prev_counts

        if stochastic:
            # Add small random noise to counts
            noise = torch.randint(-1, 2, transfer.shape, device=transfer.device)
            transfer = (transfer + noise).clamp(0)

        return transfer


class RecursiveDiffusionLM(nn.Module):
    """A diffusion language model built on top of a TinyRecursiveModel.

    Uses masked diffusion (MDLM) with the recursive TRM as the denoising backbone.

    Args:
        backbone: A TinyRecursiveModel instance
        mask_token_id: Token ID for [MASK]
        noise_schedule: Noise schedule for diffusion
    """

    def __init__(
        self,
        backbone,
        mask_token_id: int,
        noise_schedule: NoiseSchedule | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.mask_token_id = mask_token_id
        self.noise_schedule = noise_schedule or NoiseSchedule()
        self.vocab_size = backbone.vocab_size

    @property
    def device(self):
        return self.backbone.device

    # ── Forward diffusion process ────────────────────────────────────

    def _apply_mask(
        self,
        x0: Tensor,
        t: Tensor,
        eligible_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply masking noise to clean sequences.

        Args:
            x0: (batch, seq_len) — clean token IDs
            t: (batch,) — timesteps in [0, 1]
            eligible_mask: (batch, seq_len) — True where tokens may be masked
        Returns:
            x_t: (batch, seq_len) — masked sequences
            mask: (batch, seq_len) — True = masked position
        """
        batch, seq_len = x0.shape
        device = x0.device

        # Each token is independently masked with probability = t
        mask_prob = self.noise_schedule.mask_prob(t)  # (batch,)
        # Expand to per-position masking probabilities
        mask_prob = mask_prob.unsqueeze(-1).expand(-1, seq_len)  # (batch, seq_len)

        # Sample mask decisions
        random_mask = torch.rand(batch, seq_len, device=device)
        is_masked = random_mask < mask_prob  # (batch, seq_len)
        if eligible_mask is not None:
            eligible_mask = eligible_mask.bool()
            is_masked = is_masked & eligible_mask
            for row in range(batch):
                if eligible_mask[row].any() and not is_masked[row].any():
                    first_eligible = torch.nonzero(eligible_mask[row], as_tuple=False)[0, 0]
                    is_masked[row, first_eligible] = True

        # Apply mask: replace masked positions with mask_token_id
        x_t = torch.where(is_masked, torch.full_like(x0, self.mask_token_id), x0)

        return x_t, is_masked

    # ── Training step ────────────────────────────────────────────────

    def forward(
        self,
        x0: Tensor,
        padding_mask: Tensor | None = None,
        loss_mask: Tensor | None = None,
    ) -> dict:
        """Training step: mask, predict, compute loss.

        Args:
            x0: (batch, seq_len) — clean token sequences
            padding_mask: (batch, seq_len) — 1 = valid, 0 = padding
            loss_mask: (batch, seq_len) — 1 = positions eligible for masking/loss
        Returns:
            dict with loss, token_loss, logits, and diagnostics
        """
        batch, _seq_len = x0.shape
        device = x0.device

        # 1. Sample timesteps
        t = self.noise_schedule.sample_t(batch, device=device)  # (batch,)

        # 2. Apply masking
        eligible_mask = loss_mask.bool() if loss_mask is not None else None
        x_t, is_masked = self._apply_mask(x0, t, eligible_mask=eligible_mask)

        # 3. Forward pass through backbone with timestep conditioning
        out = self.backbone(
            tokens=x_t,
            timestep=t,
            labels=x0,
            mask=padding_mask,
            mask_token_id=self.mask_token_id,
        )

        # 4. Compute diffusion loss: only on masked positions
        logits = out["logits"]  # (batch, seq_len, vocab)
        loss_all = functional.cross_entropy(
            logits.permute(0, 2, 1),  # (batch, vocab, seq_len)
            x0,
            reduction="none",
        )  # (batch, seq_len)

        # Only compute loss at masked positions
        # Don't count padding tokens.
        effective_mask = is_masked & padding_mask.bool() if padding_mask is not None else is_masked
        if loss_mask is not None:
            effective_mask = effective_mask & loss_mask.bool()

        n_masked = effective_mask.sum()
        if n_masked > 0:
            diffusion_loss = (loss_all * effective_mask.float()).sum() / n_masked
        else:
            diffusion_loss = torch.tensor(0.0, device=device)

        # Also track accuracy on masked positions
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct_on_masked = ((preds == x0) & effective_mask).float().sum()
            masked_acc = correct_on_masked / n_masked if n_masked > 0 else torch.tensor(0.0)

        return {
            "loss": diffusion_loss,
            "token_loss": diffusion_loss,
            "logits": logits,
            "t": t,
            "masked_ratio": is_masked.float().mean(),
            "masked_acc": masked_acc,
            "n_masked": n_masked,
        }

    # ── Inference (generation) ───────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        prompt: Tensor | None = None,
        max_new_tokens: int = 128,
        steps: int = 128,
        block_size: int = 128,
        temperature: float = 0.0,
        return_history: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Generate text using iterative masked diffusion.

        The coarsest-to-finest generation: starts fully masked, then
        progressively reveals the highest-confidence tokens.

        Args:
            prompt: (batch, prompt_len) or None — optional prompt tokens
            max_new_tokens: number of new tokens to generate
            steps: number of diffusion steps (more = better quality)
            block_size: generate in blocks of this size (for long sequences)
            temperature: sampling temperature (0 = greedy argmax)
            return_history: if True, also return intermediate sequences
        Returns:
            If return_history: (final_sequences, list_of_intermediates)
            Else: final_sequences
        """
        device = self.device

        # ---- Setup ----
        if prompt is not None:
            batch, prompt_len = prompt.shape
            max_len = prompt_len + max_new_tokens
        else:
            batch = 1
            prompt_len = 0
            max_len = max_new_tokens

        # Initialize sequence: prompt (if given) + [MASK] * max_new_tokens
        x = torch.full((batch, max_len), self.mask_token_id, dtype=torch.long, device=device)
        if prompt is not None:
            x[:, :prompt_len] = prompt

        # Track which positions are "given" (prompt) vs "masked" (to generate)
        given = torch.zeros((batch, max_len), dtype=torch.bool, device=device)
        if prompt is not None:
            given[:, :prompt_len] = True

        # Attention mask: valid positions up to max_new_tokens
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long, device=device)
        max_valid = prompt_len + max_new_tokens
        if max_valid > max_len:
            max_valid = max_len
        attention_mask[:, :max_valid] = 1

        # Block scheduling: partition the generated region into blocks
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = math.ceil(steps / num_blocks)

        histories = [x.clone()] if return_history else None

        for block_idx in range(num_blocks):
            # ---- Build mask index for this block ----
            block_start = prompt_len + block_idx * block_size
            block_end = min(block_start + block_size, max_len)
            block_width = block_end - block_start

            if block_width <= 0:
                continue

            # Which positions in this block are still masked?
            block_mask = x[:, block_start:block_end] == self.mask_token_id

            # How many tokens to reveal per step in this block?
            num_transfer = self.noise_schedule.get_num_transfer_tokens(
                block_mask, steps_per_block
            )  # (batch, steps_per_block)
            effective_steps = num_transfer.size(1)

            for step_i in range(effective_steps):
                # ---- Forward pass ----
                # Use the current block's timestep (from the schedule)
                # We approximate the "noise level" of the current state
                remaining_masks = (x == self.mask_token_id).float().sum(dim=-1)  # (batch,)
                total_to_generate = torch.full_like(remaining_masks, max_new_tokens)
                # Estimate t = fraction of positions still masked
                t_est = (remaining_masks / total_to_generate.clamp(min=1)).clamp(0, 1)
                # But also clamp to a minimum noise level
                t_est = t_est.clamp(min=0.01, max=0.99)

                out = self.backbone(
                    tokens=x,
                    timestep=t_est,
                    mask=attention_mask,
                    mask_token_id=self.mask_token_id,
                )
                logits = out["logits"]  # (batch, seq_len, vocab)

                # ---- Predict clean tokens ----
                # Add Gumbel noise for stochastic sampling
                if temperature > 0:
                    logits = logits + _gumbel_noise(logits.shape, device=device) * temperature

                x0_pred = logits.argmax(dim=-1)  # (batch, seq_len)

                # ---- Compute confidence for each prediction ----
                probs = functional.softmax(logits, dim=-1)
                confidence = torch.gather(
                    probs, dim=-1, index=x0_pred.unsqueeze(-1)
                ).squeeze(-1)  # (batch, seq_len)

                # ---- Select which tokens to reveal ----
                # Only consider masked positions in the current block
                # Set confidence to -inf for already-revealed positions
                select_confidence = confidence.clone()
                # Mask out: positions outside this block
                select_confidence[:, :block_start] = -float("inf")
                select_confidence[:, block_end:] = -float("inf")
                # Mask out: positions already given (prompt)
                select_confidence[given] = -float("inf")
                # Mask out: positions already revealed in previous steps
                select_confidence[x != self.mask_token_id] = -float("inf")

                # Pick the top-k most confident positions for each sample
                n_to_reveal = num_transfer[:, step_i]  # (batch,)
                transfer = torch.zeros_like(x, dtype=torch.bool)

                for b in range(batch):
                    k = int(n_to_reveal[b].item())
                    if k > 0:
                        # Get the k highest confidence scores for this sample
                        _vals, idxs = torch.topk(select_confidence[b], k=k)
                        transfer[b, idxs] = True

                # Reveal: put the predicted tokens in the selected positions
                x = torch.where(transfer, x0_pred, x)

                if histories is not None:
                    histories.append(x.clone())

        if return_history:
            return x, histories
        return x


def _gumbel_noise(shape, device="cpu"):
    """Sample from Gumbel(0, 1) distribution."""
    noise = torch.rand(shape, device=device)
    return -(-noise.log()).log()


def train_diffusion_step(
    model: RecursiveDiffusionLM,
    x0: Tensor,
    optimizer: torch.optim.Optimizer,
    padding_mask: Tensor | None = None,
    clip_grad_norm: float = 1.0,
) -> dict:
    """Single training step for the recursive diffusion LM.

    Args:
        model: RecursiveDiffusionLM instance
        x0: (batch, seq_len) — clean sequences
        optimizer: PyTorch optimizer
        padding_mask: (batch, seq_len) — 1 = valid, 0 = padding
        clip_grad_norm: max gradient norm for clipping
    Returns:
        dict with loss and diagnostics
    """
    optimizer.zero_grad()
    out = model(x0, padding_mask=padding_mask)
    out["loss"].backward()
    if clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    optimizer.step()
    return out


if __name__ == "__main__":
    # Quick test
    from rdlm.trm import TinyRecursiveModel

    print("Testing RecursiveDiffusionLM...")

    # Create backbone
    backbone = TinyRecursiveModel(
        vocab_size=100,
        dim=64,
        num_heads=2,
        max_seq_len=64,
        num_latent_refinements=4,
        num_refinement_blocks=2,
    )

    # Create diffusion LM
    diff_lm = RecursiveDiffusionLM(
        backbone=backbone,
        mask_token_id=99,  # last token is [MASK]
    )

    # Test training step
    batch, seq_len = 4, 16
    x0 = torch.randint(0, 98, (batch, seq_len))  # random clean tokens
    out = diff_lm(x0)
    print(
        f"Training step: loss={out['loss'].item():.4f}, "
        f"masked_ratio={out['masked_ratio'].item():.3f}"
    )

    # Test generation
    prompt = torch.randint(0, 98, (1, 4))
    generated = diff_lm.sample(prompt=prompt, max_new_tokens=16, steps=32, block_size=16)
    print(f"Generation: prompt (1, 4) → output {generated.shape}")
    # Count how many mask tokens are left
    n_masks = (generated == 99).sum().item()
    print(f"  Remaining [MASK] tokens: {n_masks}/{generated.numel()}")
    print("✅ RecursiveDiffusionLM works!")
