"""Grid-native ARC encoder and output diffusion models."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from rdlm.diffusion_lm import NoiseSchedule
from rdlm.tiny_block import TinyBlock

PAD_COLOR_ID = 0
MASK_COLOR_ID = 11
NUM_COLOR_TOKENS = 12
NUM_OUTPUT_COLORS = 10


class ArcStructuredEncoder(nn.Module):
    """Embed ARC grid cells with color, position, role, and example identity."""

    def __init__(
        self,
        dim: int,
        max_grid_size: int = 30,
        max_examples: int = 8,
        num_roles: int = 4,
    ):
        super().__init__()
        self.color_embed = nn.Embedding(NUM_COLOR_TOKENS, dim)
        self.row_embed = nn.Embedding(max_grid_size, dim)
        self.col_embed = nn.Embedding(max_grid_size, dim)
        self.role_embed = nn.Embedding(num_roles, dim)
        self.example_embed = nn.Embedding(max_examples + 1, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        colors: Tensor,
        rows: Tensor,
        cols: Tensor,
        roles: Tensor,
        examples: Tensor,
    ) -> Tensor:
        return self.norm(
            self.color_embed(colors + 1)
            + self.row_embed(rows)
            + self.col_embed(cols)
            + self.role_embed(roles)
            + self.example_embed(examples)
        )


class ArcOutputDiffusion(nn.Module):
    """Denoise query output grid cells conditioned on encoded ARC context."""

    def __init__(
        self,
        dim: int = 256,
        max_grid_size: int = 30,
        max_examples: int = 8,
        num_heads: int = 4,
        num_latent_refinements: int = 6,
        num_refinement_blocks: int = 2,
        noise_schedule: NoiseSchedule | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.noise_schedule = noise_schedule or NoiseSchedule()
        self.encoder = ArcStructuredEncoder(
            dim=dim,
            max_grid_size=max_grid_size,
            max_examples=max_examples,
        )
        self.target_color_embed = nn.Embedding(NUM_COLOR_TOKENS, dim)
        self.target_row_embed = nn.Embedding(max_grid_size, dim)
        self.target_col_embed = nn.Embedding(max_grid_size, dim)
        self.target_role_embed = nn.Embedding(1, dim)
        self.time_embed = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )
        self.output_init = nn.Parameter(torch.randn(dim) * 0.02)
        self.latent_init = nn.Parameter(torch.randn(dim) * 0.02)
        self.network = TinyBlock(dim=dim, num_heads=num_heads)
        self.to_logits = nn.Linear(dim, NUM_OUTPUT_COLORS, bias=False)
        self.num_latent_refinements = num_latent_refinements
        self.num_refinement_blocks = num_refinement_blocks

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _time_embedding(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=t.device).float()
            * torch.log(torch.tensor(10000.0, device=t.device))
            / max(half_dim - 1, 1)
        )
        angles = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.time_embed(emb)

    def _mask_targets(
        self,
        target_colors: Tensor,
        target_mask: Tensor,
        t: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, target_len = target_colors.shape
        mask_prob = self.noise_schedule.mask_prob(t).unsqueeze(-1).expand(-1, target_len)
        is_masked = torch.rand(batch, target_len, device=target_colors.device) < mask_prob
        is_masked = is_masked & target_mask.bool()
        for row in range(batch):
            if target_mask[row].any() and not is_masked[row].any():
                first_target = torch.nonzero(target_mask[row], as_tuple=False)[0, 0]
                is_masked[row, first_target] = True
        noisy = torch.where(
            is_masked,
            torch.full_like(target_colors, MASK_COLOR_ID - 1),
            target_colors,
        )
        return noisy, is_masked

    def _target_inputs(self, colors: Tensor, rows: Tensor, cols: Tensor, t: Tensor) -> Tensor:
        batch, target_len = colors.shape
        role = torch.zeros((batch, target_len), dtype=torch.long, device=colors.device)
        time_emb = self._time_embedding(t).unsqueeze(1)
        return (
            self.target_color_embed(colors + 1)
            + self.target_row_embed(rows)
            + self.target_col_embed(cols)
            + self.target_role_embed(role)
            + time_emb
        )

    def _refine(self, inputs: Tensor, attention_mask: Tensor) -> Tensor:
        batch, seq_len, _dim = inputs.shape
        outputs = self.output_init.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
        latents = self.latent_init.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
        for block_idx in range(self.num_refinement_blocks):
            context = torch.no_grad if block_idx < self.num_refinement_blocks - 1 else nullcontext
            with context():
                for _ in range(self.num_latent_refinements):
                    latents = self.network(outputs + latents + inputs, mask=attention_mask)
                outputs = self.network(outputs + latents, mask=attention_mask)
        return outputs

    def forward(
        self,
        context_colors: Tensor,
        context_rows: Tensor,
        context_cols: Tensor,
        context_roles: Tensor,
        context_examples: Tensor,
        context_mask: Tensor,
        target_colors: Tensor,
        target_rows: Tensor,
        target_cols: Tensor,
        target_mask: Tensor,
    ) -> dict[str, Tensor]:
        batch = target_colors.shape[0]
        t = self.noise_schedule.sample_t(batch, device=str(target_colors.device))
        noisy_targets, is_masked = self._mask_targets(target_colors, target_mask, t)

        context_inputs = self.encoder(
            context_colors,
            context_rows,
            context_cols,
            context_roles,
            context_examples,
        )
        target_inputs = self._target_inputs(noisy_targets, target_rows, target_cols, t)
        inputs = torch.cat([context_inputs, target_inputs], dim=1)
        attention_mask = torch.cat([context_mask, target_mask], dim=1)
        outputs = self._refine(inputs, attention_mask)
        target_outputs = outputs[:, context_inputs.shape[1] :]
        logits = self.to_logits(target_outputs)
        loss_all = functional.cross_entropy(
            logits.permute(0, 2, 1),
            target_colors,
            reduction="none",
        )
        effective_mask = is_masked & target_mask.bool()
        n_masked = effective_mask.sum()
        loss = (loss_all * effective_mask.float()).sum() / n_masked
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = ((preds == target_colors) & effective_mask).float().sum()
            masked_acc = correct / n_masked
        return {
            "loss": loss,
            "logits": logits,
            "masked_acc": masked_acc,
            "masked_ratio": is_masked.float().mean(),
            "n_masked": n_masked,
        }

    @torch.no_grad()
    def sample(
        self,
        context_colors: Tensor,
        context_rows: Tensor,
        context_cols: Tensor,
        context_roles: Tensor,
        context_examples: Tensor,
        context_mask: Tensor,
        target_rows: Tensor,
        target_cols: Tensor,
        target_mask: Tensor,
        steps: int = 64,
    ) -> Tensor:
        batch, target_len = target_rows.shape
        current = torch.full(
            (batch, target_len),
            MASK_COLOR_ID - 1,
            dtype=torch.long,
            device=self.device,
        )
        revealed = torch.zeros((batch, target_len), dtype=torch.bool, device=self.device)
        transfer_schedule = self.noise_schedule.get_num_transfer_tokens(target_mask.bool(), steps)

        context_inputs = self.encoder(
            context_colors,
            context_rows,
            context_cols,
            context_roles,
            context_examples,
        )
        for step_idx in range(transfer_schedule.shape[1]):
            remaining = (target_mask.bool() & ~revealed).float().sum(dim=-1)
            total = target_mask.float().sum(dim=-1).clamp(min=1)
            t = (remaining / total).clamp(min=0.01, max=0.99)
            target_inputs = self._target_inputs(current, target_rows, target_cols, t)
            inputs = torch.cat([context_inputs, target_inputs], dim=1)
            attention_mask = torch.cat([context_mask, target_mask], dim=1)
            outputs = self._refine(inputs, attention_mask)
            logits = self.to_logits(outputs[:, context_inputs.shape[1] :])
            preds = logits.argmax(dim=-1)
            probs = functional.softmax(logits, dim=-1)
            confidence = torch.gather(probs, -1, preds.unsqueeze(-1)).squeeze(-1)
            confidence = confidence.masked_fill(~target_mask.bool() | revealed, -float("inf"))
            transfer = torch.zeros_like(revealed)
            for row in range(batch):
                k = int(transfer_schedule[row, step_idx].item())
                if k > 0:
                    _vals, idxs = torch.topk(confidence[row], k=min(k, target_len))
                    transfer[row, idxs] = True
            current = torch.where(transfer, preds, current)
            revealed = revealed | transfer
        return torch.where(target_mask.bool(), current, torch.zeros_like(current))
