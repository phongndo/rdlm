"""Grid-native ARC encoder and output diffusion models."""

from __future__ import annotations

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
    ):
        super().__init__()
        if not 0.0 <= stochastic_depth_prob < 1.0:
            raise ValueError("stochastic_depth_prob must be in [0, 1)")
        if aux_loss_weight < 0.0:
            raise ValueError("aux_loss_weight must be non-negative")
        self.dim = dim
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
            "aux_loss": aux_loss,
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
    ) -> tuple[Tensor, Tensor]:
        batch, target_len = target_rows.shape
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
        for step_idx in range(transfer_schedule.shape[1]):
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
            confidence = confidence.masked_fill(~target_mask.bool() | revealed, -float("inf"))
            transfer = torch.zeros_like(revealed)
            for row in range(batch):
                k = int(transfer_schedule[row, step_idx].item())
                if k > 0:
                    _vals, idxs = torch.topk(confidence[row], k=min(k, target_len))
                    transfer[row, idxs] = True
            current = torch.where(transfer, preds, current)
            token_log_probs = torch.where(transfer, clean_pred_log_probs, token_log_probs)
            revealed = revealed | transfer
        return (
            torch.where(target_mask.bool(), current, torch.zeros_like(current)),
            torch.where(target_mask.bool(), token_log_probs, torch.zeros_like(token_log_probs)),
        )

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
    ) -> dict[str, Tensor]:
        batch, target_len = target_rows.shape
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

        revealed_history: list[Tensor] = []
        sample_history: list[Tensor] = []
        confidence_history: list[Tensor] = []
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
            selection_confidence = confidence.masked_fill(
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
    ) -> tuple[Tensor, Tensor]:
        if num_candidates < 1:
            raise ValueError("num_candidates must be at least 1")
        if temperature_start < 0 or temperature_end < 0:
            raise ValueError("temperatures must be non-negative")
        if strategy not in {"confidence", "majority"}:
            raise ValueError("strategy must be 'confidence' or 'majority'")

        samples: list[Tensor] = []
        log_probs: list[Tensor] = []
        for _ in range(num_candidates):
            sample, scores = self._sample_with_scores(
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
            )
            samples.append(sample)
            log_probs.append(scores)

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
            return (
                torch.where(target_mask_bool, best, torch.zeros_like(best)),
                torch.where(target_mask_bool, best_scores, torch.zeros_like(best_scores)),
            )

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
        return (
            torch.where(target_mask_bool, voted, torch.zeros_like(voted)),
            torch.where(target_mask_bool, voted_scores, torch.zeros_like(voted_scores)),
        )

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
        )
        return samples
