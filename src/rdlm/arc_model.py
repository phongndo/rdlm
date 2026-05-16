"""Grid-native ARC encoder and output diffusion models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from rdlm.diffusion_lm import NoiseSchedule
from rdlm.tiny_block import TinyBlock

PAD_COLOR_ID = 0
MASK_COLOR_ID = 11
NUM_COLOR_TOKENS = 12
NUM_OUTPUT_COLORS = 10


@dataclass(frozen=True)
class ShapeCandidate:
    """One proposed ARC output shape."""

    height: int
    width: int
    log_prob: float
    source: str


@dataclass(frozen=True)
class StructuredCandidate:
    """One generated structured ARC output candidate."""

    height: int
    width: int
    sample: Tensor
    token_log_probs: Tensor
    mean_token_log_prob: float
    shape_log_prob: float
    source: str
    mean_confidence: float = 0.0
    mean_entropy: float = 0.0
    temporal_consistency: float = 0.0
    steps_used: int = 0


class ArcStructuredEncoder(nn.Module):
    """Embed ARC grid cells with color, position, role, and example identity."""

    def __init__(
        self,
        dim: int,
        max_grid_size: int = 30,
        max_examples: int = 8,
        num_roles: int = 4,
        use_object_features: bool = False,
    ):
        super().__init__()
        self.use_object_features = use_object_features
        self.color_embed = nn.Embedding(NUM_COLOR_TOKENS, dim)
        self.row_embed = nn.Embedding(max_grid_size, dim)
        self.col_embed = nn.Embedding(max_grid_size, dim)
        self.role_embed = nn.Embedding(num_roles, dim)
        self.example_embed = nn.Embedding(max_examples + 1, dim)
        if use_object_features:
            self.object_id_embed = nn.Embedding(max_grid_size * max_grid_size + 1, dim)
            self.object_size_embed = nn.Embedding(max_grid_size * max_grid_size + 1, dim)
            self.object_height_embed = nn.Embedding(max_grid_size + 1, dim)
            self.object_width_embed = nn.Embedding(max_grid_size + 1, dim)
            self.object_rel_row_embed = nn.Embedding(max_grid_size + 1, dim)
            self.object_rel_col_embed = nn.Embedding(max_grid_size + 1, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        colors: Tensor,
        rows: Tensor,
        cols: Tensor,
        roles: Tensor,
        examples: Tensor,
        object_ids: Tensor | None = None,
        object_size_buckets: Tensor | None = None,
        object_heights: Tensor | None = None,
        object_widths: Tensor | None = None,
        object_rel_rows: Tensor | None = None,
        object_rel_cols: Tensor | None = None,
    ) -> Tensor:
        encoded = (
            self.color_embed(colors + 1)
            + self.row_embed(rows)
            + self.col_embed(cols)
            + self.role_embed(roles)
            + self.example_embed(examples)
        )
        if self.use_object_features:
            if (
                object_ids is None
                or object_size_buckets is None
                or object_heights is None
                or object_widths is None
                or object_rel_rows is None
                or object_rel_cols is None
            ):
                raise ValueError(
                    "object feature tensors are required when use_object_features=True"
                )
            encoded = (
                encoded
                + self.object_id_embed(object_ids)
                + self.object_size_embed(object_size_buckets)
                + self.object_height_embed(object_heights)
                + self.object_width_embed(object_widths)
                + self.object_rel_row_embed(object_rel_rows)
                + self.object_rel_col_embed(object_rel_cols)
            )
        return self.norm(encoded)


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
        gradient_checkpointing: bool = False,
        stochastic_depth_prob: float = 0.0,
        aux_loss_weight: float = 0.0,
        use_object_features: bool = False,
        use_shape_head: bool = False,
        shape_loss_weight: float = 0.1,
    ):
        super().__init__()
        if not 0.0 <= stochastic_depth_prob < 1.0:
            raise ValueError("stochastic_depth_prob must be in [0, 1)")
        if aux_loss_weight < 0.0:
            raise ValueError("aux_loss_weight must be non-negative")
        if shape_loss_weight < 0.0:
            raise ValueError("shape_loss_weight must be non-negative")
        self.dim = dim
        self.max_grid_size = max_grid_size
        self.noise_schedule = noise_schedule or NoiseSchedule()
        self.encoder = ArcStructuredEncoder(
            dim=dim,
            max_grid_size=max_grid_size,
            max_examples=max_examples,
            use_object_features=use_object_features,
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
        self.gradient_checkpointing = gradient_checkpointing
        self.stochastic_depth_prob = stochastic_depth_prob
        self.aux_loss_weight = aux_loss_weight
        self.use_object_features = use_object_features
        self.use_shape_head = use_shape_head
        self.shape_loss_weight = shape_loss_weight
        if use_shape_head:
            self.shape_height_head = nn.Linear(dim, max_grid_size)
            self.shape_width_head = nn.Linear(dim, max_grid_size)

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

    def _encode_context(
        self,
        context_colors: Tensor,
        context_rows: Tensor,
        context_cols: Tensor,
        context_roles: Tensor,
        context_examples: Tensor,
        context_object_ids: Tensor | None = None,
        context_object_size_buckets: Tensor | None = None,
        context_object_heights: Tensor | None = None,
        context_object_widths: Tensor | None = None,
        context_object_rel_rows: Tensor | None = None,
        context_object_rel_cols: Tensor | None = None,
    ) -> Tensor:
        return self.encoder(
            context_colors,
            context_rows,
            context_cols,
            context_roles,
            context_examples,
            object_ids=context_object_ids,
            object_size_buckets=context_object_size_buckets,
            object_heights=context_object_heights,
            object_widths=context_object_widths,
            object_rel_rows=context_object_rel_rows,
            object_rel_cols=context_object_rel_cols,
        )

    def _pool_context(self, context_inputs: Tensor, context_mask: Tensor) -> Tensor:
        mask = context_mask.bool().unsqueeze(-1)
        denom = mask.float().sum(dim=1).clamp(min=1.0)
        return (context_inputs * mask.float()).sum(dim=1) / denom

    def _shape_logits_from_context(
        self,
        context_inputs: Tensor,
        context_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self.use_shape_head:
            raise ValueError("shape head is disabled")
        pooled = self._pool_context(context_inputs, context_mask)
        return self.shape_height_head(pooled), self.shape_width_head(pooled)

    @staticmethod
    def _target_grid_tensors(
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        rows = torch.arange(height, device=device).repeat_interleave(width).unsqueeze(0)
        cols = torch.arange(width, device=device).repeat(height).unsqueeze(0)
        mask = torch.ones((1, height * width), dtype=torch.bool, device=device)
        return rows.long(), cols.long(), mask

    @staticmethod
    @torch.no_grad()
    def _dependency_score(
        confidence: Tensor,
        preds: Tensor,
        target_mask: Tensor,
        target_rows: Tensor,
        target_cols: Tensor,
    ) -> Tensor:
        """DOS: Adjust confidence based on spatial dependency consistency.

        For each cell in a grid, computes what fraction of its 4-connected neighbors
        predict the same color. Adjusts confidence by (1 + 0.5 * neighbor_consistency).
        Cells spatially consistent with their neighbors get a boost; isolated cells don't.
        This is training-free and grid-native.

        Args:
            confidence: (batch, seq_len) — per-token confidence scores
            preds: (batch, seq_len) — predicted token IDs
            target_mask: (batch, seq_len) — True for valid target positions
            target_rows: (batch, seq_len) — row coordinate of each cell (row-major order)
            target_cols: (batch, seq_len) — col coordinate of each cell (row-major order)
        Returns:
            Adjusted confidence of same shape.
        """
        batch, _seq_len = preds.shape
        device = preds.device
        neighbor_scores = torch.zeros_like(confidence)

        for b in range(batch):
            mask = target_mask[b]
            if not mask.any():
                continue
            rows = target_rows[b][mask]
            cols = target_cols[b][mask]
            height = int(rows.max().item()) + 1
            width = int(cols.max().item()) + 1
            total_cells = height * width

            # Extract predictions for this sample, ordered by their flat index
            # target_rows/target_cols are in row-major order, so flat index = r * width + c
            # but we must account for possible padding via the mask
            valid_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)  # (n_valid,)
            grid_preds = torch.full((total_cells,), -1, dtype=torch.long, device=device)
            for i, idx in enumerate(valid_indices):
                r = int(rows[i].item())
                c = int(cols[i].item())
                flat = r * width + c
                grid_preds[flat] = preds[b, idx]
            grid_preds_2d = grid_preds.view(height, width)
            has_value = grid_preds_2d >= 0

            # Vectorized neighbor consistency: for each cell, check up/down/left/right
            neighbor_count = torch.zeros((height, width), device=device)
            neighbor_match = torch.zeros((height, width), device=device)

            # Right neighbors (use float multiplication for logical AND)
            if width > 1:
                rc = has_value[:, :-1] & has_value[:, 1:]
                rc_f = rc.float()
                neighbor_count[:, :-1] += rc_f
                neighbor_count[:, 1:] += rc_f
                color_match_rc = (grid_preds_2d[:, :-1] == grid_preds_2d[:, 1:]).float()
                neighbor_match[:, :-1] += rc_f * color_match_rc
                neighbor_match[:, 1:] += rc_f * color_match_rc

            # Down neighbors
            if height > 1:
                dc = has_value[:-1, :] & has_value[1:, :]
                dc_f = dc.float()
                neighbor_count[:-1, :] += dc_f
                neighbor_count[1:, :] += dc_f
                color_match_dc = (grid_preds_2d[:-1, :] == grid_preds_2d[1:, :]).float()
                neighbor_match[:-1, :] += dc_f * color_match_dc
                neighbor_match[1:, :] += dc_f * color_match_dc

            # Consistency per cell
            consistency = torch.where(
                neighbor_count > 0,
                neighbor_match / neighbor_count,
                torch.zeros_like(neighbor_match),
            )
            consistency = consistency * has_value.float()

            # Map back to flat batch indices
            flat_consistency = consistency.flatten()  # (total_cells,) row-major
            for i, idx in enumerate(valid_indices):
                r = int(rows[i].item())
                c = int(cols[i].item())
                flat = r * width + c
                neighbor_scores[b, idx] = flat_consistency[flat]

        # Boost confidence by up to 50% based on neighbor consistency
        return confidence * (1.0 + 0.5 * neighbor_scores)

    @staticmethod
    @torch.no_grad()
    def _structured_order_score(
        target_mask: Tensor,
        target_rows: Tensor,
        target_cols: Tensor,
        strategy: str,
    ) -> Tensor:
        """Return deterministic reveal priorities for ARC grid positions."""
        scores = torch.zeros(target_rows.shape, dtype=torch.float, device=target_rows.device)
        mask_bool = target_mask.bool()
        for batch_idx in range(target_rows.shape[0]):
            mask = mask_bool[batch_idx]
            if not mask.any():
                continue
            rows = target_rows[batch_idx].float()
            cols = target_cols[batch_idx].float()
            valid_rows = rows[mask]
            valid_cols = cols[mask]
            height = valid_rows.max() + 1.0
            width = valid_cols.max() + 1.0
            flat_tiebreak = (rows * width + cols) / (height * width).clamp(min=1.0)
            if strategy == "scanline":
                priority = -flat_tiebreak
            elif strategy == "border-first":
                distance_to_border = torch.minimum(
                    torch.minimum(rows, cols),
                    torch.minimum(height - 1.0 - rows, width - 1.0 - cols),
                )
                priority = -distance_to_border - 1e-4 * flat_tiebreak
            elif strategy == "center-first":
                center_row = (height - 1.0) / 2.0
                center_col = (width - 1.0) / 2.0
                distance_to_center = (rows - center_row).abs() + (cols - center_col).abs()
                priority = -distance_to_center - 1e-4 * flat_tiebreak
            else:
                raise ValueError(f"unknown structured order strategy: {strategy}")
            scores[batch_idx] = priority
        return scores

    @staticmethod
    @torch.no_grad()
    def _temporal_vote(
        final_preds: Tensor,
        final_log_probs: Tensor,
        pred_history: list[Tensor],
        target_mask: Tensor,
        weight: str = "uniform",
    ) -> tuple[Tensor, Tensor]:
        """Post-process: vote across prediction history to recover lost correct answers.

        The "Time Is a Feature" paper (arxiv 2508.09138) shows that correct answers
        often emerge in the middle of the denoising process but get overwritten in
        later steps. This method votes across intermediate predictions.

        Args:
            final_preds: (batch, seq_len) — predictions from the last denoising step
            final_log_probs: (batch, seq_len) — log probs from the last step
            pred_history: list of (batch, seq_len) — predictions recorded at each step
            target_mask: (batch, seq_len) — True for valid target positions
            weight: vote weighting — "uniform" (majority) or "recency" (later steps weighted higher)
        Returns:
            (voted_preds, voted_log_probs) with same shapes.
        """
        if not pred_history:
            return final_preds, final_log_probs

        _batch, _seq_len = final_preds.shape
        device = final_preds.device
        n_steps = len(pred_history)
        mask = target_mask.bool()

        # Stack: (steps, batch, seq_len)
        history = torch.stack(pred_history, dim=0)

        # One-hot encode votes: (steps, batch, seq_len, vocab)
        one_hot = functional.one_hot(history, num_classes=NUM_OUTPUT_COLORS).float()

        if weight == "recency":
            step_weights = torch.arange(1, n_steps + 1, device=device).float()
            step_weights = step_weights / step_weights.sum()
            step_weights = step_weights.view(-1, 1, 1, 1)
            counts = (one_hot * step_weights).sum(dim=0)
        else:  # uniform
            counts = one_hot.sum(dim=0)  # (batch, seq_len, vocab)

        voted = counts.argmax(dim=-1)  # (batch, seq_len)

        # Confidence = vote fraction
        max_counts = counts.max(dim=-1).values  # (batch, seq_len)
        vote_fraction = max_counts / n_steps if weight == "uniform" else max_counts
        vote_fraction = vote_fraction.clamp(min=1e-6)
        voted_log_probs = torch.log(vote_fraction)

        # Only apply to valid positions, zero out padding
        voted = torch.where(mask, voted, torch.zeros_like(voted))
        voted_log_probs = torch.where(mask, voted_log_probs, final_log_probs)

        return voted, voted_log_probs

    @staticmethod
    @torch.no_grad()
    def _temporal_consistency(
        final_preds: Tensor,
        pred_history: list[Tensor],
        target_mask: Tensor,
    ) -> Tensor:
        """Return per-example agreement between final predictions and history."""
        if not pred_history:
            return torch.ones(final_preds.shape[0], device=final_preds.device)
        mask = target_mask.bool()
        history = torch.stack(pred_history, dim=0)
        matches = (history == final_preds.unsqueeze(0)) & mask.unsqueeze(0)
        denom = mask.float().sum(dim=-1).clamp(min=1.0) * len(pred_history)
        return matches.float().sum(dim=(0, 2)) / denom

    @staticmethod
    def _adaptive_steps(
        steps: int,
        target_mask: Tensor,
        adaptive_sample_steps: bool,
        min_sample_steps: int,
        max_sample_steps: int | None,
        reference_grid_area: int,
    ) -> int:
        if not adaptive_sample_steps:
            return steps
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if min_sample_steps < 1:
            raise ValueError("min_sample_steps must be at least 1")
        if max_sample_steps is not None and max_sample_steps < min_sample_steps:
            raise ValueError("max_sample_steps must be >= min_sample_steps")
        reference = max(reference_grid_area, 1)
        area = float(target_mask.bool().sum(dim=-1).float().mean().item())
        scaled = round(steps * (area / reference) ** 0.5)
        upper = max_sample_steps if max_sample_steps is not None else steps
        return max(min_sample_steps, min(max(scaled, 1), upper))

    @staticmethod
    def _candidate_diagnostics(
        token_log_probs: Tensor,
        target_mask: Tensor,
        temporal_consistency: Tensor,
        steps_used: int,
    ) -> dict[str, Tensor | int]:
        mask = target_mask.bool()
        denom = target_mask.float().sum(dim=-1).clamp(min=1.0)
        mean_confidence = (token_log_probs.exp() * mask.float()).sum(dim=-1) / denom
        return {
            "mean_entropy": torch.zeros_like(mean_confidence),
            "mean_confidence": mean_confidence,
            "temporal_consistency": temporal_consistency,
            "steps_used": steps_used,
        }

    def _refine_one_block(
        self,
        outputs: Tensor,
        latents: Tensor,
        inputs: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        for _ in range(self.num_latent_refinements):
            latents = self.network(outputs + latents + inputs, mask=attention_mask)
        outputs = self.network(outputs + latents, mask=attention_mask)
        return outputs, latents

    def _should_skip_refinement_block(self, block_idx: int) -> bool:
        if not self.training or self.stochastic_depth_prob == 0.0:
            return False
        if block_idx == self.num_refinement_blocks - 1:
            return False
        return bool(torch.rand((), device=self.device) < self.stochastic_depth_prob)

    def _refine(
        self,
        inputs: Tensor,
        attention_mask: Tensor,
        return_intermediate: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        batch, seq_len, _dim = inputs.shape
        outputs = self.output_init.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
        latents = self.latent_init.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
        intermediates: list[Tensor] = []
        for block_idx in range(self.num_refinement_blocks):
            if self._should_skip_refinement_block(block_idx):
                if return_intermediate:
                    intermediates.append(outputs)
                continue
            if self.gradient_checkpointing and self.training:
                outputs, latents = checkpoint(
                    self._refine_one_block,
                    outputs,
                    latents,
                    inputs,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                outputs, latents = self._refine_one_block(outputs, latents, inputs, attention_mask)
            if return_intermediate:
                intermediates.append(outputs)
        if return_intermediate:
            return outputs, intermediates
        return outputs

    def _masked_loss(self, logits: Tensor, target_colors: Tensor, effective_mask: Tensor) -> Tensor:
        loss_all = functional.cross_entropy(
            logits.permute(0, 2, 1),
            target_colors,
            reduction="none",
        )
        n_masked = effective_mask.sum()
        return (loss_all * effective_mask.float()).sum() / n_masked

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
        context_object_ids: Tensor | None = None,
        context_object_size_buckets: Tensor | None = None,
        context_object_heights: Tensor | None = None,
        context_object_widths: Tensor | None = None,
        context_object_rel_rows: Tensor | None = None,
        context_object_rel_cols: Tensor | None = None,
        target_heights: Tensor | None = None,
        target_widths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch = target_colors.shape[0]
        t = self.noise_schedule.sample_t(batch, device=str(target_colors.device))
        noisy_targets, is_masked = self._mask_targets(target_colors, target_mask, t)

        context_inputs = self._encode_context(
            context_colors=context_colors,
            context_rows=context_rows,
            context_cols=context_cols,
            context_roles=context_roles,
            context_examples=context_examples,
            context_object_ids=context_object_ids,
            context_object_size_buckets=context_object_size_buckets,
            context_object_heights=context_object_heights,
            context_object_widths=context_object_widths,
            context_object_rel_rows=context_object_rel_rows,
            context_object_rel_cols=context_object_rel_cols,
        )
        target_inputs = self._target_inputs(noisy_targets, target_rows, target_cols, t)
        inputs = torch.cat([context_inputs, target_inputs], dim=1)
        attention_mask = torch.cat([context_mask, target_mask], dim=1)
        refined = self._refine(
            inputs,
            attention_mask,
            return_intermediate=self.aux_loss_weight > 0.0,
        )
        if self.aux_loss_weight > 0.0:
            outputs, intermediates = refined
        else:
            outputs = refined
            intermediates = []
        target_outputs = outputs[:, context_inputs.shape[1] :]
        logits = self.to_logits(target_outputs)
        effective_mask = is_masked & target_mask.bool()
        n_masked = effective_mask.sum()
        loss = self._masked_loss(logits, target_colors, effective_mask)
        aux_loss = torch.zeros((), device=target_colors.device)
        shape_loss = torch.zeros((), device=target_colors.device)
        shape_height_logits = torch.empty(0, device=target_colors.device)
        shape_width_logits = torch.empty(0, device=target_colors.device)
        if self.aux_loss_weight > 0.0 and len(intermediates) > 1:
            aux_terms = [
                self._masked_loss(
                    self.to_logits(block_outputs[:, context_inputs.shape[1] :]),
                    target_colors,
                    effective_mask,
                )
                for block_outputs in intermediates[:-1]
            ]
            aux_loss = torch.stack(aux_terms).mean()
            loss = loss + self.aux_loss_weight * aux_loss
        if self.use_shape_head:
            if target_heights is None or target_widths is None:
                raise ValueError(
                    "target_heights and target_widths are required with use_shape_head=True"
                )
            shape_height_logits, shape_width_logits = self._shape_logits_from_context(
                context_inputs,
                context_mask,
            )
            height_labels = (target_heights - 1).clamp(min=0, max=self.max_grid_size - 1)
            width_labels = (target_widths - 1).clamp(min=0, max=self.max_grid_size - 1)
            shape_loss = (
                functional.cross_entropy(shape_height_logits, height_labels)
                + functional.cross_entropy(shape_width_logits, width_labels)
            ) * 0.5
            loss = loss + self.shape_loss_weight * shape_loss
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = ((preds == target_colors) & effective_mask).float().sum()
            masked_acc = correct / n_masked
            if self.use_shape_head:
                height_acc = (shape_height_logits.argmax(dim=-1) == height_labels).float().mean()
                width_acc = (shape_width_logits.argmax(dim=-1) == width_labels).float().mean()
                shape_acc = (height_acc + width_acc) * 0.5
            else:
                height_acc = torch.zeros((), device=target_colors.device)
                width_acc = torch.zeros((), device=target_colors.device)
                shape_acc = torch.zeros((), device=target_colors.device)
        return {
            "loss": loss,
            "logits": logits,
            "masked_acc": masked_acc,
            "masked_ratio": is_masked.float().mean(),
            "n_masked": n_masked,
            "aux_loss": aux_loss,
            "shape_loss": shape_loss,
            "shape_height_logits": shape_height_logits,
            "shape_width_logits": shape_width_logits,
            "shape_height_acc": height_acc,
            "shape_width_acc": width_acc,
            "shape_acc": shape_acc,
        }

    @torch.no_grad()
    def _sample_with_scores(
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
        temperature_start: float = 0.0,
        temperature_end: float = 0.0,
        context_object_ids: Tensor | None = None,
        context_object_size_buckets: Tensor | None = None,
        context_object_heights: Tensor | None = None,
        context_object_widths: Tensor | None = None,
        context_object_rel_rows: Tensor | None = None,
        context_object_rel_cols: Tensor | None = None,
        # ── Training-free sampling strategy improvements ────────────
        sampling_strategy: str = "confidence",
        enable_calibration: bool = False,
        calibration_strength: float = 0.3,
        temporal_vote: bool = False,
        temporal_vote_weight: str = "uniform",
        vote_start_ratio: float = 0.0,
        adaptive_sample_steps: bool = False,
        min_sample_steps: int = 8,
        max_sample_steps: int | None = None,
        reference_grid_area: int = 100,
        early_stop_confidence: float | None = None,
        early_stop_entropy: float | None = None,
        early_stop_patience: int = 1,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor | float | int]]:
        batch, target_len = target_rows.shape
        steps = self._adaptive_steps(
            steps,
            target_mask,
            adaptive_sample_steps,
            min_sample_steps,
            max_sample_steps,
            reference_grid_area,
        )
        current = torch.full(
            (batch, target_len),
            MASK_COLOR_ID - 1,
            dtype=torch.long,
            device=self.device,
        )
        token_log_probs = torch.zeros((batch, target_len), dtype=torch.float, device=self.device)
        revealed = torch.zeros((batch, target_len), dtype=torch.bool, device=self.device)
        transfer_schedule = self.noise_schedule.get_num_transfer_tokens(target_mask.bool(), steps)

        context_inputs = self._encode_context(
            context_colors=context_colors,
            context_rows=context_rows,
            context_cols=context_cols,
            context_roles=context_roles,
            context_examples=context_examples,
            context_object_ids=context_object_ids,
            context_object_size_buckets=context_object_size_buckets,
            context_object_heights=context_object_heights,
            context_object_widths=context_object_widths,
            context_object_rel_rows=context_object_rel_rows,
            context_object_rel_cols=context_object_rel_cols,
        )

        # ── Temporal voting state ───────────────────────────────────
        pred_history: list[Tensor] = []
        vote_start_step = int(vote_start_ratio * transfer_schedule.shape[1]) if temporal_vote else 0

        # ── Calibration state ───────────────────────────────────────
        prev_preds: Tensor | None = None
        flip_counts: Tensor = torch.zeros((batch, target_len), device=self.device)
        stable_steps = 0
        last_entropy = torch.zeros((batch, target_len), dtype=torch.float, device=self.device)
        steps_used = 0

        for step_idx in range(transfer_schedule.shape[1]):
            steps_used = step_idx + 1
            if transfer_schedule.shape[1] <= 1:
                temperature = temperature_end
            else:
                progress = step_idx / (transfer_schedule.shape[1] - 1)
                temperature = temperature_start + (temperature_end - temperature_start) * progress
            remaining = (target_mask.bool() & ~revealed).float().sum(dim=-1)
            total = target_mask.float().sum(dim=-1).clamp(min=1)
            t = (remaining / total).clamp(min=0.01, max=0.99)
            target_inputs = self._target_inputs(current, target_rows, target_cols, t)
            inputs = torch.cat([context_inputs, target_inputs], dim=1)
            attention_mask = torch.cat([context_mask, target_mask], dim=1)
            outputs = self._refine(inputs, attention_mask)
            logits = self.to_logits(outputs[:, context_inputs.shape[1] :])
            clean_log_probs = functional.log_softmax(logits, dim=-1)
            if temperature > 0:
                sample_probs = functional.softmax(logits / temperature, dim=-1)
                preds = torch.multinomial(
                    sample_probs.reshape(-1, NUM_OUTPUT_COLORS),
                    num_samples=1,
                ).reshape(batch, target_len)
            else:
                preds = logits.argmax(dim=-1)
            clean_pred_log_probs = torch.gather(
                clean_log_probs,
                -1,
                preds.unsqueeze(-1),
            ).squeeze(-1)
            confidence = clean_pred_log_probs.exp()
            probs = clean_log_probs.exp()
            entropy = -(probs * clean_log_probs).sum(dim=-1)
            last_entropy = entropy

            # ── Record prediction history for temporal voting ─────────
            if temporal_vote and step_idx >= vote_start_step:
                pred_history.append(preds.clone())

            # ── Calibration: penalize confidence for flip-prone positions ──
            if enable_calibration:
                if prev_preds is not None:
                    masked_positions = ~revealed
                    flip_events = (preds != prev_preds) & masked_positions
                    flip_counts = flip_counts + flip_events
                prev_preds = preds.clone()
                if step_idx > 1:
                    flip_rate = flip_counts / (step_idx + 1)
                    calibration_factor = 1.0 - calibration_strength * flip_rate
                    confidence = confidence * calibration_factor

            # ── Token reveal priority ──────────────────────────────────
            if sampling_strategy == "dos":
                selection_confidence = self._dependency_score(
                    confidence, preds, target_mask, target_rows, target_cols
                )
            elif sampling_strategy in {"scanline", "border-first", "center-first"}:
                selection_confidence = self._structured_order_score(
                    target_mask, target_rows, target_cols, sampling_strategy
                )
            else:
                selection_confidence = confidence

            selection_confidence = selection_confidence.masked_fill(
                ~target_mask.bool() | revealed,
                -float("inf"),
            )
            transfer = torch.zeros_like(revealed)
            for row in range(batch):
                k = int(transfer_schedule[row, step_idx].item())
                if k > 0:
                    _vals, idxs = torch.topk(selection_confidence[row], k=min(k, target_len))
                    transfer[row, idxs] = True
            current = torch.where(transfer, preds, current)
            token_log_probs = torch.where(transfer, clean_pred_log_probs, token_log_probs)
            revealed = revealed | transfer

            if bool((target_mask.bool() & ~revealed).any()):
                remaining_mask = target_mask.bool() & ~revealed
                confident = True
                low_entropy = True
                if early_stop_confidence is not None:
                    remaining_conf = confidence[remaining_mask]
                    confident = bool(
                        remaining_conf.numel()
                        and remaining_conf.min().item() >= early_stop_confidence
                    )
                if early_stop_entropy is not None:
                    remaining_entropy = entropy[remaining_mask]
                    low_entropy = bool(
                        remaining_entropy.numel()
                        and remaining_entropy.max().item() <= early_stop_entropy
                    )
                if (early_stop_confidence is not None or early_stop_entropy is not None) and (
                    confident and low_entropy
                ):
                    stable_steps += 1
                else:
                    stable_steps = 0
                if stable_steps >= max(early_stop_patience, 1):
                    current = torch.where(remaining_mask, preds, current)
                    token_log_probs = torch.where(
                        remaining_mask,
                        clean_pred_log_probs,
                        token_log_probs,
                    )
                    revealed = revealed | remaining_mask
                    break
            elif bool((target_mask.bool() & revealed).all()):
                break

        # ── Temporal voting post-processing ──────────────────────────
        temporal_consistency = self._temporal_consistency(current, pred_history, target_mask)
        if temporal_vote and pred_history:
            current, token_log_probs = self._temporal_vote(
                current, token_log_probs, pred_history, target_mask,
                weight=temporal_vote_weight,
            )

        samples = torch.where(target_mask.bool(), current, torch.zeros_like(current))
        scores = torch.where(target_mask.bool(), token_log_probs, torch.zeros_like(token_log_probs))
        if return_diagnostics:
            valid_entropy = torch.where(
                target_mask.bool(),
                last_entropy,
                torch.zeros_like(last_entropy),
            )
            denom = target_mask.float().sum(dim=-1).clamp(min=1.0)
            diagnostics: dict[str, Tensor | float | int] = {
                "mean_entropy": (valid_entropy * target_mask.float()).sum(dim=-1) / denom,
                "mean_confidence": (scores.exp() * target_mask.float()).sum(dim=-1) / denom,
                "temporal_consistency": temporal_consistency,
                "steps_used": steps_used,
            }
            return samples, scores, diagnostics
        return (
            samples,
            scores,
        )

    @torch.no_grad()
    def predict_shapes(
        self,
        context_colors: Tensor,
        context_rows: Tensor,
        context_cols: Tensor,
        context_roles: Tensor,
        context_examples: Tensor,
        context_mask: Tensor,
        top_k: int = 5,
        context_object_ids: Tensor | None = None,
        context_object_size_buckets: Tensor | None = None,
        context_object_heights: Tensor | None = None,
        context_object_widths: Tensor | None = None,
        context_object_rel_rows: Tensor | None = None,
        context_object_rel_cols: Tensor | None = None,
    ) -> list[list[ShapeCandidate]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.use_shape_head:
            return [[] for _ in range(context_colors.shape[0])]
        context_inputs = self._encode_context(
            context_colors=context_colors,
            context_rows=context_rows,
            context_cols=context_cols,
            context_roles=context_roles,
            context_examples=context_examples,
            context_object_ids=context_object_ids,
            context_object_size_buckets=context_object_size_buckets,
            context_object_heights=context_object_heights,
            context_object_widths=context_object_widths,
            context_object_rel_rows=context_object_rel_rows,
            context_object_rel_cols=context_object_rel_cols,
        )
        height_logits, width_logits = self._shape_logits_from_context(context_inputs, context_mask)
        height_log_probs = functional.log_softmax(height_logits, dim=-1)
        width_log_probs = functional.log_softmax(width_logits, dim=-1)
        k = min(top_k, self.max_grid_size)
        height_scores, height_idxs = torch.topk(height_log_probs, k=k, dim=-1)
        width_scores, width_idxs = torch.topk(width_log_probs, k=k, dim=-1)
        batch_candidates: list[list[ShapeCandidate]] = []
        for batch_idx in range(context_colors.shape[0]):
            candidates: list[ShapeCandidate] = []
            for height_rank in range(k):
                for width_rank in range(k):
                    height = int(height_idxs[batch_idx, height_rank].item()) + 1
                    width = int(width_idxs[batch_idx, width_rank].item()) + 1
                    log_prob = float(
                        height_scores[batch_idx, height_rank].item()
                        + width_scores[batch_idx, width_rank].item()
                    )
                    candidates.append(
                        ShapeCandidate(
                            height=height,
                            width=width,
                            log_prob=log_prob,
                            source="shape_head",
                        )
                    )
            candidates.sort(key=lambda candidate: candidate.log_prob, reverse=True)
            batch_candidates.append(candidates[:top_k])
        return batch_candidates

    @torch.no_grad()
    def sample_candidates(
        self,
        context_colors: Tensor,
        context_rows: Tensor,
        context_cols: Tensor,
        context_roles: Tensor,
        context_examples: Tensor,
        context_mask: Tensor,
        candidate_shapes: list[ShapeCandidate],
        steps: int = 64,
        inference_mode: str = "greedy",
        num_candidates: int = 8,
        temperature_start: float = 1.0,
        temperature_end: float = 0.1,
        ensemble_strategy: str = "confidence",
        context_object_ids: Tensor | None = None,
        context_object_size_buckets: Tensor | None = None,
        context_object_heights: Tensor | None = None,
        context_object_widths: Tensor | None = None,
        context_object_rel_rows: Tensor | None = None,
        context_object_rel_cols: Tensor | None = None,
        # ── Sampling strategy improvements ──────────────────────────
        sampling_strategy: str = "confidence",
        enable_calibration: bool = False,
        calibration_strength: float = 0.3,
        temporal_vote: bool = False,
        temporal_vote_weight: str = "uniform",
        vote_start_ratio: float = 0.0,
        adaptive_sample_steps: bool = False,
        min_sample_steps: int = 8,
        max_sample_steps: int | None = None,
        reference_grid_area: int = 100,
        early_stop_confidence: float | None = None,
        early_stop_entropy: float | None = None,
        early_stop_patience: int = 1,
    ) -> list[StructuredCandidate]:
        if context_colors.shape[0] != 1:
            raise ValueError("sample_candidates currently supports batch size 1")
        if inference_mode not in {"greedy", "ensemble"}:
            raise ValueError("inference_mode must be 'greedy' or 'ensemble'")

        candidates: list[StructuredCandidate] = []
        for shape in candidate_shapes:
            target_rows, target_cols, target_mask = self._target_grid_tensors(
                shape.height,
                shape.width,
                self.device,
            )
            if inference_mode == "greedy":
                sample_result = self._sample_with_scores(
                    context_colors=context_colors,
                    context_rows=context_rows,
                    context_cols=context_cols,
                    context_roles=context_roles,
                    context_examples=context_examples,
                    context_mask=context_mask,
                    target_rows=target_rows,
                    target_cols=target_cols,
                    target_mask=target_mask,
                    steps=steps,
                    context_object_ids=context_object_ids,
                    context_object_size_buckets=context_object_size_buckets,
                    context_object_heights=context_object_heights,
                    context_object_widths=context_object_widths,
                    context_object_rel_rows=context_object_rel_rows,
                    context_object_rel_cols=context_object_rel_cols,
                    sampling_strategy=sampling_strategy,
                    enable_calibration=enable_calibration,
                    calibration_strength=calibration_strength,
                    temporal_vote=temporal_vote,
                    temporal_vote_weight=temporal_vote_weight,
                    vote_start_ratio=vote_start_ratio,
                    adaptive_sample_steps=adaptive_sample_steps,
                    min_sample_steps=min_sample_steps,
                    max_sample_steps=max_sample_steps,
                    reference_grid_area=reference_grid_area,
                    early_stop_confidence=early_stop_confidence,
                    early_stop_entropy=early_stop_entropy,
                    early_stop_patience=early_stop_patience,
                    return_diagnostics=True,
                )
                if len(sample_result) == 3:
                    sample, token_log_probs, diagnostics = sample_result
                else:
                    sample, token_log_probs = sample_result
                    diagnostics = self._candidate_diagnostics(
                        token_log_probs,
                        target_mask,
                        torch.ones((1,), device=self.device),
                        steps,
                    )
            else:
                sample_result = self._sample_ensemble_with_scores(
                    context_colors=context_colors,
                    context_rows=context_rows,
                    context_cols=context_cols,
                    context_roles=context_roles,
                    context_examples=context_examples,
                    context_mask=context_mask,
                    target_rows=target_rows,
                    target_cols=target_cols,
                    target_mask=target_mask,
                    steps=steps,
                    num_candidates=num_candidates,
                    temperature_start=temperature_start,
                    temperature_end=temperature_end,
                    strategy=ensemble_strategy,
                    context_object_ids=context_object_ids,
                    context_object_size_buckets=context_object_size_buckets,
                    context_object_heights=context_object_heights,
                    context_object_widths=context_object_widths,
                    context_object_rel_rows=context_object_rel_rows,
                    context_object_rel_cols=context_object_rel_cols,
                    sampling_strategy=sampling_strategy,
                    enable_calibration=enable_calibration,
                    calibration_strength=calibration_strength,
                    temporal_vote=temporal_vote,
                    temporal_vote_weight=temporal_vote_weight,
                    vote_start_ratio=vote_start_ratio,
                    adaptive_sample_steps=adaptive_sample_steps,
                    min_sample_steps=min_sample_steps,
                    max_sample_steps=max_sample_steps,
                    reference_grid_area=reference_grid_area,
                    early_stop_confidence=early_stop_confidence,
                    early_stop_entropy=early_stop_entropy,
                    early_stop_patience=early_stop_patience,
                    return_diagnostics=True,
                )
                if len(sample_result) == 3:
                    sample, token_log_probs, diagnostics = sample_result
                else:
                    sample, token_log_probs = sample_result
                    diagnostics = self._candidate_diagnostics(
                        token_log_probs,
                        target_mask,
                        torch.ones((1,), device=self.device),
                        steps,
                    )
            valid_scores = token_log_probs[0][target_mask[0]]
            mean_log_prob = float(valid_scores.mean().item()) if valid_scores.numel() else 0.0
            mean_confidence = diagnostics["mean_confidence"]
            mean_entropy = diagnostics["mean_entropy"]
            temporal_consistency = diagnostics["temporal_consistency"]
            candidates.append(
                StructuredCandidate(
                    height=shape.height,
                    width=shape.width,
                    sample=sample[0],
                    token_log_probs=token_log_probs[0],
                    mean_token_log_prob=mean_log_prob,
                    shape_log_prob=shape.log_prob,
                    source=shape.source,
                    mean_confidence=float(mean_confidence[0].item())
                    if isinstance(mean_confidence, Tensor)
                    else float(mean_confidence),
                    mean_entropy=float(mean_entropy[0].item())
                    if isinstance(mean_entropy, Tensor)
                    else float(mean_entropy),
                    temporal_consistency=float(temporal_consistency[0].item())
                    if isinstance(temporal_consistency, Tensor)
                    else float(temporal_consistency),
                    steps_used=int(diagnostics["steps_used"]),
                )
            )
        return candidates

    @torch.no_grad()
    def sample_with_trace(
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
        context_object_ids: Tensor | None = None,
        context_object_size_buckets: Tensor | None = None,
        context_object_heights: Tensor | None = None,
        context_object_widths: Tensor | None = None,
        context_object_rel_rows: Tensor | None = None,
        context_object_rel_cols: Tensor | None = None,
        # ── Sampling strategy improvements ──────────────────────────
        sampling_strategy: str = "confidence",
        enable_calibration: bool = False,
        calibration_strength: float = 0.3,
        temporal_vote: bool = False,
        temporal_vote_weight: str = "uniform",
        vote_start_ratio: float = 0.0,
        adaptive_sample_steps: bool = False,
        min_sample_steps: int = 8,
        max_sample_steps: int | None = None,
        reference_grid_area: int = 100,
        early_stop_confidence: float | None = None,
        early_stop_entropy: float | None = None,
        early_stop_patience: int = 1,
    ) -> dict[str, Tensor]:
        batch, target_len = target_rows.shape
        steps = self._adaptive_steps(
            steps,
            target_mask,
            adaptive_sample_steps,
            min_sample_steps,
            max_sample_steps,
            reference_grid_area,
        )
        current = torch.full(
            (batch, target_len),
            MASK_COLOR_ID - 1,
            dtype=torch.long,
            device=self.device,
        )
        token_log_probs = torch.zeros((batch, target_len), dtype=torch.float, device=self.device)
        revealed = torch.zeros((batch, target_len), dtype=torch.bool, device=self.device)
        target_mask_bool = target_mask.bool()
        transfer_schedule = self.noise_schedule.get_num_transfer_tokens(target_mask_bool, steps)
        context_inputs = self._encode_context(
            context_colors=context_colors,
            context_rows=context_rows,
            context_cols=context_cols,
            context_roles=context_roles,
            context_examples=context_examples,
            context_object_ids=context_object_ids,
            context_object_size_buckets=context_object_size_buckets,
            context_object_heights=context_object_heights,
            context_object_widths=context_object_widths,
            context_object_rel_rows=context_object_rel_rows,
            context_object_rel_cols=context_object_rel_cols,
        )

        # ── Temporal voting state ───────────────────────────────────
        pred_history: list[Tensor] = []
        vote_start_step = int(vote_start_ratio * transfer_schedule.shape[1]) if temporal_vote else 0

        # ── Calibration state ───────────────────────────────────────
        prev_preds: Tensor | None = None
        flip_counts: Tensor = torch.zeros((batch, target_len), device=self.device)

        revealed_history: list[Tensor] = []
        sample_history: list[Tensor] = []
        confidence_history: list[Tensor] = []
        stable_steps = 0
        for step_idx in range(transfer_schedule.shape[1]):
            remaining = (target_mask_bool & ~revealed).float().sum(dim=-1)
            total = target_mask.float().sum(dim=-1).clamp(min=1)
            t = (remaining / total).clamp(min=0.01, max=0.99)
            target_inputs = self._target_inputs(current, target_rows, target_cols, t)
            inputs = torch.cat([context_inputs, target_inputs], dim=1)
            attention_mask = torch.cat([context_mask, target_mask], dim=1)
            outputs = self._refine(inputs, attention_mask)
            logits = self.to_logits(outputs[:, context_inputs.shape[1] :])
            clean_log_probs = functional.log_softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            clean_pred_log_probs = torch.gather(
                clean_log_probs,
                -1,
                preds.unsqueeze(-1),
            ).squeeze(-1)
            confidence = clean_pred_log_probs.exp()

            # ── Record prediction history for temporal voting ─────────
            if temporal_vote and step_idx >= vote_start_step:
                pred_history.append(preds.clone())

            # ── Calibration: penalize confidence for flip-prone positions ──
            if enable_calibration:
                if prev_preds is not None:
                    masked_positions = ~revealed
                    flip_events = (preds != prev_preds) & masked_positions
                    flip_counts = flip_counts + flip_events
                prev_preds = preds.clone()
                if step_idx > 1:
                    flip_rate = flip_counts / (step_idx + 1)
                    calibration_factor = 1.0 - calibration_strength * flip_rate
                    confidence = confidence * calibration_factor

            # ── Token reveal priority ──────────────────────────────────
            if sampling_strategy == "dos":
                selection_confidence = self._dependency_score(
                    confidence, preds, target_mask, target_rows, target_cols
                )
            elif sampling_strategy in {"scanline", "border-first", "center-first"}:
                selection_confidence = self._structured_order_score(
                    target_mask, target_rows, target_cols, sampling_strategy
                )
            else:
                selection_confidence = confidence

            selection_confidence = selection_confidence.masked_fill(
                ~target_mask_bool | revealed,
                -float("inf"),
            )
            transfer = torch.zeros_like(revealed)
            for row in range(batch):
                k = int(transfer_schedule[row, step_idx].item())
                if k > 0:
                    _vals, idxs = torch.topk(selection_confidence[row], k=min(k, target_len))
                    transfer[row, idxs] = True
            current = torch.where(transfer, preds, current)
            token_log_probs = torch.where(transfer, clean_pred_log_probs, token_log_probs)
            revealed = revealed | transfer
            sample_history.append(torch.where(target_mask_bool, current, torch.zeros_like(current)))
            revealed_history.append(revealed.clone())
            confidence_history.append(
                torch.where(target_mask_bool, confidence, torch.zeros_like(confidence))
            )
            if bool((target_mask_bool & ~revealed).any()):
                remaining_mask = target_mask_bool & ~revealed
                confident = True
                low_entropy = True
                if early_stop_confidence is not None:
                    remaining_conf = confidence[remaining_mask]
                    confident = bool(
                        remaining_conf.numel()
                        and remaining_conf.min().item() >= early_stop_confidence
                    )
                if early_stop_entropy is not None:
                    remaining_entropy = (-(clean_log_probs.exp() * clean_log_probs).sum(dim=-1))[
                        remaining_mask
                    ]
                    low_entropy = bool(
                        remaining_entropy.numel()
                        and remaining_entropy.max().item() <= early_stop_entropy
                    )
                if (early_stop_confidence is not None or early_stop_entropy is not None) and (
                    confident and low_entropy
                ):
                    stable_steps += 1
                else:
                    stable_steps = 0
                if stable_steps >= max(early_stop_patience, 1):
                    current = torch.where(remaining_mask, preds, current)
                    token_log_probs = torch.where(
                        remaining_mask,
                        clean_pred_log_probs,
                        token_log_probs,
                    )
                    revealed = revealed | remaining_mask
                    sample_history[-1] = torch.where(
                        target_mask_bool,
                        current,
                        torch.zeros_like(current),
                    )
                    revealed_history[-1] = revealed.clone()
                    break
            elif bool((target_mask_bool & revealed).all()):
                break

        # ── Temporal voting post-processing ──────────────────────────
        if temporal_vote and pred_history:
            current, token_log_probs = self._temporal_vote(
                current, token_log_probs, pred_history, target_mask,
                weight=temporal_vote_weight,
            )

        history_shape = (batch, 0, target_len)
        return {
            "samples": torch.where(target_mask_bool, current, torch.zeros_like(current)),
            "token_log_probs": torch.where(
                target_mask_bool,
                token_log_probs,
                torch.zeros_like(token_log_probs),
            ),
            "revealed_history": torch.stack(revealed_history, dim=1)
            if revealed_history
            else torch.zeros(history_shape, dtype=torch.bool, device=self.device),
            "sample_history": torch.stack(sample_history, dim=1)
            if sample_history
            else torch.zeros(history_shape, dtype=torch.long, device=self.device),
            "confidence_history": torch.stack(confidence_history, dim=1)
            if confidence_history
            else torch.zeros(history_shape, dtype=torch.float, device=self.device),
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
        context_object_ids: Tensor | None = None,
        context_object_size_buckets: Tensor | None = None,
        context_object_heights: Tensor | None = None,
        context_object_widths: Tensor | None = None,
        context_object_rel_rows: Tensor | None = None,
        context_object_rel_cols: Tensor | None = None,
        # ── Sampling strategy improvements ──────────────────────────
        sampling_strategy: str = "confidence",
        enable_calibration: bool = False,
        calibration_strength: float = 0.3,
        temporal_vote: bool = False,
        temporal_vote_weight: str = "uniform",
        vote_start_ratio: float = 0.0,
        adaptive_sample_steps: bool = False,
        min_sample_steps: int = 8,
        max_sample_steps: int | None = None,
        reference_grid_area: int = 100,
        early_stop_confidence: float | None = None,
        early_stop_entropy: float | None = None,
        early_stop_patience: int = 1,
    ) -> Tensor:
        samples, _scores = self._sample_with_scores(
            context_colors=context_colors,
            context_rows=context_rows,
            context_cols=context_cols,
            context_roles=context_roles,
            context_examples=context_examples,
            context_mask=context_mask,
            target_rows=target_rows,
            target_cols=target_cols,
            target_mask=target_mask,
            steps=steps,
            context_object_ids=context_object_ids,
            context_object_size_buckets=context_object_size_buckets,
            context_object_heights=context_object_heights,
            context_object_widths=context_object_widths,
            context_object_rel_rows=context_object_rel_rows,
            context_object_rel_cols=context_object_rel_cols,
            sampling_strategy=sampling_strategy,
            enable_calibration=enable_calibration,
            calibration_strength=calibration_strength,
            temporal_vote=temporal_vote,
            temporal_vote_weight=temporal_vote_weight,
            vote_start_ratio=vote_start_ratio,
            adaptive_sample_steps=adaptive_sample_steps,
            min_sample_steps=min_sample_steps,
            max_sample_steps=max_sample_steps,
            reference_grid_area=reference_grid_area,
            early_stop_confidence=early_stop_confidence,
            early_stop_entropy=early_stop_entropy,
            early_stop_patience=early_stop_patience,
        )
        return samples

    @torch.no_grad()
    def _sample_ensemble_with_scores(
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
        num_candidates: int = 8,
        temperature_start: float = 1.0,
        temperature_end: float = 0.1,
        strategy: str = "confidence",
        context_object_ids: Tensor | None = None,
        context_object_size_buckets: Tensor | None = None,
        context_object_heights: Tensor | None = None,
        context_object_widths: Tensor | None = None,
        context_object_rel_rows: Tensor | None = None,
        context_object_rel_cols: Tensor | None = None,
        # ── Sampling strategy improvements ──────────────────────────
        sampling_strategy: str = "confidence",
        enable_calibration: bool = False,
        calibration_strength: float = 0.3,
        temporal_vote: bool = False,
        temporal_vote_weight: str = "uniform",
        vote_start_ratio: float = 0.0,
        adaptive_sample_steps: bool = False,
        min_sample_steps: int = 8,
        max_sample_steps: int | None = None,
        reference_grid_area: int = 100,
        early_stop_confidence: float | None = None,
        early_stop_entropy: float | None = None,
        early_stop_patience: int = 1,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor | float | int]]:
        if num_candidates < 1:
            raise ValueError("num_candidates must be at least 1")
        if temperature_start < 0 or temperature_end < 0:
            raise ValueError("temperatures must be non-negative")
        if strategy not in {"confidence", "majority"}:
            raise ValueError("strategy must be 'confidence' or 'majority'")

        samples: list[Tensor] = []
        log_probs: list[Tensor] = []
        diagnostics_list: list[dict[str, Tensor | float | int]] = []
        for _ in range(num_candidates):
            sample_result = self._sample_with_scores(
                context_colors=context_colors,
                context_rows=context_rows,
                context_cols=context_cols,
                context_roles=context_roles,
                context_examples=context_examples,
                context_mask=context_mask,
                target_rows=target_rows,
                target_cols=target_cols,
                target_mask=target_mask,
                steps=steps,
                temperature_start=temperature_start,
                temperature_end=temperature_end,
                context_object_ids=context_object_ids,
                context_object_size_buckets=context_object_size_buckets,
                context_object_heights=context_object_heights,
                context_object_widths=context_object_widths,
                context_object_rel_rows=context_object_rel_rows,
                context_object_rel_cols=context_object_rel_cols,
                sampling_strategy=sampling_strategy,
                enable_calibration=enable_calibration,
                calibration_strength=calibration_strength,
                temporal_vote=temporal_vote,
                temporal_vote_weight=temporal_vote_weight,
                vote_start_ratio=vote_start_ratio,
                adaptive_sample_steps=adaptive_sample_steps,
                min_sample_steps=min_sample_steps,
                max_sample_steps=max_sample_steps,
                reference_grid_area=reference_grid_area,
                early_stop_confidence=early_stop_confidence,
                early_stop_entropy=early_stop_entropy,
                early_stop_patience=early_stop_patience,
                return_diagnostics=True,
            )
            if len(sample_result) == 3:
                sample, scores, diagnostics = sample_result
            else:
                sample, scores = sample_result
                diagnostics = self._candidate_diagnostics(
                    scores,
                    target_mask,
                    torch.ones((target_rows.shape[0],), device=self.device),
                    steps,
                )
            samples.append(sample)
            log_probs.append(scores)
            diagnostics_list.append(diagnostics)

        sample_stack = torch.stack(samples, dim=0)
        log_prob_stack = torch.stack(log_probs, dim=0)
        target_mask_bool = target_mask.bool()

        if strategy == "confidence":
            denom = target_mask.float().sum(dim=-1).clamp(min=1)
            candidate_scores = (log_prob_stack * target_mask_bool.unsqueeze(0)).sum(dim=-1) / denom
            best_idx = candidate_scores.argmax(dim=0)
            batch_idx = torch.arange(target_rows.shape[0], device=self.device)
            best = sample_stack[best_idx, batch_idx]
            best_scores = log_prob_stack[best_idx, batch_idx]
            result = (
                torch.where(target_mask_bool, best, torch.zeros_like(best)),
                torch.where(target_mask_bool, best_scores, torch.zeros_like(best_scores)),
            )
            if return_diagnostics:
                gathered = {
                    "mean_entropy": torch.stack(
                        [
                            d["mean_entropy"]
                            for d in diagnostics_list
                            if isinstance(d["mean_entropy"], Tensor)
                        ],
                        dim=0,
                    )[best_idx, batch_idx],
                    "mean_confidence": torch.stack(
                        [
                            d["mean_confidence"]
                            for d in diagnostics_list
                            if isinstance(d["mean_confidence"], Tensor)
                        ],
                        dim=0,
                    )[best_idx, batch_idx],
                    "temporal_consistency": torch.stack(
                        [
                            d["temporal_consistency"]
                            for d in diagnostics_list
                            if isinstance(d["temporal_consistency"], Tensor)
                        ],
                        dim=0,
                    )[best_idx, batch_idx],
                    "steps_used": max(int(d["steps_used"]) for d in diagnostics_list),
                }
                return result[0], result[1], gathered
            return result

        vote_counts = []
        vote_scores = []
        for color in range(NUM_OUTPUT_COLORS):
            color_matches = sample_stack == color
            vote_counts.append(color_matches.sum(dim=0))
            vote_scores.append(torch.where(color_matches, log_prob_stack, 0.0).sum(dim=0))
        counts = torch.stack(vote_counts, dim=-1)
        scores = torch.stack(vote_scores, dim=-1)
        max_counts = counts.max(dim=-1, keepdim=True).values
        tied_scores = scores.masked_fill(counts != max_counts, -float("inf"))
        voted = tied_scores.argmax(dim=-1)
        voted_scores = torch.gather(scores, -1, voted.unsqueeze(-1)).squeeze(-1)
        voted_counts = torch.gather(counts, -1, voted.unsqueeze(-1)).squeeze(-1).clamp(min=1)
        voted_scores = voted_scores / voted_counts
        result = (
            torch.where(target_mask_bool, voted, torch.zeros_like(voted)),
            torch.where(target_mask_bool, voted_scores, torch.zeros_like(voted_scores)),
        )
        if return_diagnostics:
            mean_entropy = torch.stack(
                [
                    d["mean_entropy"]
                    for d in diagnostics_list
                    if isinstance(d["mean_entropy"], Tensor)
                ],
                dim=0,
            ).mean(dim=0)
            mean_confidence = torch.stack(
                [
                    d["mean_confidence"]
                    for d in diagnostics_list
                    if isinstance(d["mean_confidence"], Tensor)
                ],
                dim=0,
            ).mean(dim=0)
            temporal_consistency = torch.stack(
                [
                    d["temporal_consistency"]
                    for d in diagnostics_list
                    if isinstance(d["temporal_consistency"], Tensor)
                ],
                dim=0,
            ).mean(dim=0)
            return result[0], result[1], {
                "mean_entropy": mean_entropy,
                "mean_confidence": mean_confidence,
                "temporal_consistency": temporal_consistency,
                "steps_used": max(int(d["steps_used"]) for d in diagnostics_list),
            }
        return result

    @torch.no_grad()
    def sample_ensemble(
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
        num_candidates: int = 8,
        temperature_start: float = 1.0,
        temperature_end: float = 0.1,
        strategy: str = "confidence",
        context_object_ids: Tensor | None = None,
        context_object_size_buckets: Tensor | None = None,
        context_object_heights: Tensor | None = None,
        context_object_widths: Tensor | None = None,
        context_object_rel_rows: Tensor | None = None,
        context_object_rel_cols: Tensor | None = None,
        # ── Sampling strategy improvements ──────────────────────────
        sampling_strategy: str = "confidence",
        enable_calibration: bool = False,
        calibration_strength: float = 0.3,
        temporal_vote: bool = False,
        temporal_vote_weight: str = "uniform",
        vote_start_ratio: float = 0.0,
        adaptive_sample_steps: bool = False,
        min_sample_steps: int = 8,
        max_sample_steps: int | None = None,
        reference_grid_area: int = 100,
        early_stop_confidence: float | None = None,
        early_stop_entropy: float | None = None,
        early_stop_patience: int = 1,
    ) -> Tensor:
        samples, _scores = self._sample_ensemble_with_scores(
            context_colors=context_colors,
            context_rows=context_rows,
            context_cols=context_cols,
            context_roles=context_roles,
            context_examples=context_examples,
            context_mask=context_mask,
            target_rows=target_rows,
            target_cols=target_cols,
            target_mask=target_mask,
            steps=steps,
            num_candidates=num_candidates,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            strategy=strategy,
            context_object_ids=context_object_ids,
            context_object_size_buckets=context_object_size_buckets,
            context_object_heights=context_object_heights,
            context_object_widths=context_object_widths,
            context_object_rel_rows=context_object_rel_rows,
            context_object_rel_cols=context_object_rel_cols,
            sampling_strategy=sampling_strategy,
            enable_calibration=enable_calibration,
            calibration_strength=calibration_strength,
            temporal_vote=temporal_vote,
            temporal_vote_weight=temporal_vote_weight,
            vote_start_ratio=vote_start_ratio,
            adaptive_sample_steps=adaptive_sample_steps,
            min_sample_steps=min_sample_steps,
            max_sample_steps=max_sample_steps,
            reference_grid_area=reference_grid_area,
            early_stop_confidence=early_stop_confidence,
            early_stop_entropy=early_stop_entropy,
            early_stop_patience=early_stop_patience,
        )
        return samples
