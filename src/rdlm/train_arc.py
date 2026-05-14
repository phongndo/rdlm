"""Train the recursive diffusion LM on serialized ARC-AGI JSON tasks."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, TypedDict

import torch
from torch.utils.data import DataLoader

from rdlm.arc import (
    MASK_TOKEN_ID,
    VOCAB_SIZE,
    ArcDataset,
    ArcStructuredDataset,
    collate_arc_examples,
    collate_structured_arc_examples,
    decode_tokens,
    find_arc_json_files,
    parse_grid,
)
from rdlm.arc_model import NUM_OUTPUT_COLORS, ArcOutputDiffusion
from rdlm.diffusion_lm import NoiseSchedule, RecursiveDiffusionLM
from rdlm.trm import TinyRecursiveModel

ArchitectureModel = RecursiveDiffusionLM | ArcOutputDiffusion


class PredictionMetrics(TypedDict):
    exact: bool
    cell_count: int
    correct_count: int
    cell_acc: float
    nonzero_iou: float
    color_hist_l1: float
    mean_token_log_prob: float
    mean_confidence: float


STRUCTURED_FORWARD_KEYS = (
    "context_colors",
    "context_rows",
    "context_cols",
    "context_roles",
    "context_examples",
    "context_mask",
    "target_colors",
    "target_rows",
    "target_cols",
    "target_mask",
    "context_object_ids",
    "context_object_size_buckets",
    "context_object_heights",
    "context_object_widths",
    "context_object_rel_rows",
    "context_object_rel_cols",
)
STRUCTURED_SAMPLE_KEYS = tuple(key for key in STRUCTURED_FORWARD_KEYS if key != "target_colors")


def choose_device(name: str) -> str:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but this PyTorch build cannot access CUDA. "
            "Install a CUDA-enabled torch build on the NVIDIA GPU host."
        )
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested, but MPS is unavailable.")
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_model(dim: int, seq_len: int) -> RecursiveDiffusionLM:
    backbone = TinyRecursiveModel(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        num_heads=4,
        max_seq_len=seq_len,
        num_latent_refinements=6,
        num_refinement_blocks=2,
    )
    return RecursiveDiffusionLM(
        backbone=backbone,
        mask_token_id=MASK_TOKEN_ID,
        noise_schedule=NoiseSchedule(),
    )


def create_structured_model(args: argparse.Namespace) -> ArcOutputDiffusion:
    return ArcOutputDiffusion(
        dim=args.dim,
        max_grid_size=args.max_grid_size,
        max_examples=args.max_examples,
        gradient_checkpointing=args.gradient_checkpointing,
        stochastic_depth_prob=args.stochastic_depth_prob,
        aux_loss_weight=args.aux_loss_weight,
        use_object_features=args.use_object_features,
    )


def save_checkpoint(
    path: Path,
    model: ArchitectureModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: ArchitectureModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint["step"]) + 1


def load_model_checkpoint(path: Path, model: ArchitectureModel, device: str) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])


def structured_forward_kwargs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    kwargs: dict[str, torch.Tensor] = {}
    for key in STRUCTURED_FORWARD_KEYS:
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            kwargs[key] = value
    return kwargs


def structured_sample_kwargs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    kwargs: dict[str, torch.Tensor] = {}
    for key in STRUCTURED_SAMPLE_KEYS:
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            kwargs[key] = value
    return kwargs


def _jsonable_args(args: argparse.Namespace | None) -> dict[str, Any]:
    if args is None:
        return {}
    jsonable: dict[str, Any] = {}
    for key, value in vars(args).items():
        jsonable[key] = str(value) if isinstance(value, Path) else value
    return jsonable


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _grid_from_values(
    values: list[int],
    shape: tuple[int, int],
    unknown_value: int | None = None,
) -> list[list[int]]:
    height, width = shape
    grid: list[list[int]] = []
    for row_idx in range(height):
        row: list[int] = []
        for col_idx in range(width):
            value = int(values[row_idx * width + col_idx])
            if unknown_value is not None and value >= NUM_OUTPUT_COLORS:
                value = unknown_value
            row.append(value)
        grid.append(row)
    return grid


def _float_grid_from_tensor(values: torch.Tensor, shape: tuple[int, int]) -> list[list[float]]:
    flat = [float(value) for value in values.detach().cpu().tolist()]
    height, width = shape
    return [flat[row_idx * width : (row_idx + 1) * width] for row_idx in range(height)]


def structured_prediction_metrics(
    generated: torch.Tensor,
    expected: torch.Tensor,
    target_mask: torch.Tensor,
    token_log_probs: torch.Tensor,
) -> PredictionMetrics:
    mask = target_mask.bool()
    predicted_values = generated[mask].long().cpu()
    expected_values = expected[mask].long().cpu()
    log_probs = token_log_probs[mask].float().cpu()
    cell_count = int(expected_values.numel())
    correct_count = int((predicted_values == expected_values).sum().item())
    predicted_nonzero = predicted_values != 0
    expected_nonzero = expected_values != 0
    nonzero_union = int((predicted_nonzero | expected_nonzero).sum().item())
    nonzero_intersection = int((predicted_nonzero & expected_nonzero).sum().item())
    predicted_hist = torch.bincount(predicted_values, minlength=NUM_OUTPUT_COLORS).float()
    expected_hist = torch.bincount(expected_values, minlength=NUM_OUTPUT_COLORS).float()
    cell_denom = max(cell_count, 1)
    return {
        "exact": bool(correct_count == cell_count),
        "cell_count": cell_count,
        "correct_count": correct_count,
        "cell_acc": correct_count / cell_denom,
        "nonzero_iou": 1.0 if nonzero_union == 0 else nonzero_intersection / nonzero_union,
        "color_hist_l1": float((predicted_hist - expected_hist).abs().sum().item() / cell_denom),
        "mean_token_log_prob": float(log_probs.mean().item()) if cell_count else 0.0,
        "mean_confidence": float(log_probs.exp().mean().item()) if cell_count else 0.0,
    }


def _structured_debug_payload(
    task_id: str,
    shape: tuple[int, int],
    expected: torch.Tensor,
    generated: torch.Tensor,
    token_log_probs: torch.Tensor,
    trace: dict[str, torch.Tensor] | None,
    metrics: PredictionMetrics,
) -> dict[str, Any]:
    expected_values = [int(value) for value in expected.detach().cpu().view(-1).tolist()]
    generated_values = [int(value) for value in generated.detach().cpu().view(-1).tolist()]
    payload: dict[str, Any] = {
        "task_id": task_id,
        "target_shape": list(shape),
        "metrics": metrics,
        "expected_grid": _grid_from_values(expected_values, shape),
        "predicted_grid": _grid_from_values(generated_values, shape),
        "token_log_prob_grid": _float_grid_from_tensor(token_log_probs.view(-1), shape),
        "confidence_grid": _float_grid_from_tensor(token_log_probs.exp().view(-1), shape),
        "trajectory": [],
        "revealed_trajectory": [],
        "confidence_trajectory": [],
    }
    if trace is None:
        return payload

    sample_history = trace["sample_history"][0].detach().cpu()
    revealed_history = trace["revealed_history"][0].detach().cpu()
    confidence_history = trace["confidence_history"][0].detach().cpu()
    payload["trajectory"] = [
        _grid_from_values([int(value) for value in step.tolist()], shape, unknown_value=-1)
        for step in sample_history
    ]
    payload["revealed_trajectory"] = [
        _grid_from_values([int(value) for value in step.tolist()], shape)
        for step in revealed_history.long()
    ]
    payload["confidence_trajectory"] = [
        _float_grid_from_tensor(step.view(-1), shape) for step in confidence_history
    ]
    return payload


@torch.no_grad()
def evaluate(
    model: RecursiveDiffusionLM,
    dataset: ArcDataset,
    device: str,
    limit: int,
    sample_steps: int,
) -> dict[str, float]:
    model.eval()
    total = min(limit, len(dataset))
    exact = 0
    valid = 0

    for idx in range(total):
        example = dataset[idx]
        prompt_len = len(example.prompt)
        prompt = torch.tensor(
            example.tokens[:prompt_len],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        expected = parse_grid(example.target)
        generated = model.sample(
            prompt=prompt,
            max_new_tokens=len(example.target),
            steps=sample_steps,
            block_size=len(example.target),
        )
        decoded = decode_tokens(generated[0, prompt_len:])
        parsed = parse_grid(decoded)
        if parsed is not None:
            valid += 1
        if parsed == expected:
            exact += 1

    model.train()
    if total == 0:
        return {"exact": 0.0, "valid": 0.0}
    return {"exact": exact / total, "valid": valid / total}


@torch.no_grad()
def evaluate_structured(
    model: ArcOutputDiffusion,
    dataset: ArcStructuredDataset,
    device: str,
    limit: int,
    sample_steps: int,
    inference_mode: str = "greedy",
    num_candidates: int = 8,
    temperature_start: float = 1.0,
    temperature_end: float = 0.1,
    ensemble_strategy: str = "confidence",
    eval_report: Path | None = None,
    debug_dir: Path | None = None,
    debug_limit: int = 5,
    args: argparse.Namespace | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    total = min(limit, len(dataset))
    exact = 0
    total_cells = 0
    correct_cells = 0
    nonzero_iou_sum = 0.0
    color_hist_l1_sum = 0.0
    token_log_prob_sum = 0.0
    confidence_sum = 0.0
    examples: list[dict[str, Any]] = []

    for idx in range(total):
        batch = collate_structured_arc_examples([dataset[idx]])
        moved = {
            key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)
        }
        sample_kwargs = structured_sample_kwargs(moved)
        trace = None
        if inference_mode == "greedy":
            trace = model.sample_with_trace(**sample_kwargs, steps=sample_steps)
            generated = trace["samples"]
            token_log_probs = trace["token_log_probs"]
        else:
            generated, token_log_probs = model._sample_ensemble_with_scores(
                **sample_kwargs,
                steps=sample_steps,
                num_candidates=num_candidates,
                temperature_start=temperature_start,
                temperature_end=temperature_end,
                strategy=ensemble_strategy,
            )
        expected = moved["target_colors"]
        target_mask = moved["target_mask"].bool()
        prediction_metrics = structured_prediction_metrics(
            generated,
            expected,
            target_mask,
            token_log_probs,
        )
        if prediction_metrics["exact"]:
            exact += 1
        cell_count = int(prediction_metrics["cell_count"])
        total_cells += cell_count
        correct_cells += int(prediction_metrics["correct_count"])
        nonzero_iou_sum += float(prediction_metrics["nonzero_iou"])
        color_hist_l1_sum += float(prediction_metrics["color_hist_l1"])
        token_log_prob_sum += float(prediction_metrics["mean_token_log_prob"]) * cell_count
        confidence_sum += float(prediction_metrics["mean_confidence"]) * cell_count

        shape = batch["target_shapes"][0]
        task_id = str(batch["task_ids"][0])
        example_row = {
            "task_id": task_id,
            "target_shape": list(shape),
            "exact": prediction_metrics["exact"],
            "cell_acc": prediction_metrics["cell_acc"],
            "nonzero_iou": prediction_metrics["nonzero_iou"],
            "color_hist_l1": prediction_metrics["color_hist_l1"],
            "mean_token_log_prob": prediction_metrics["mean_token_log_prob"],
            "mean_confidence": prediction_metrics["mean_confidence"],
        }
        examples.append(example_row)
        if debug_dir is not None and idx < debug_limit:
            debug_payload = _structured_debug_payload(
                task_id=task_id,
                shape=shape,
                expected=expected[0],
                generated=generated[0],
                token_log_probs=token_log_probs[0],
                trace=trace,
                metrics=prediction_metrics,
            )
            _write_json(debug_dir / f"{idx:04d}_{task_id}.json", debug_payload)

    model.train()
    if total == 0:
        summary = {
            "exact": 0.0,
            "valid": 1.0,
            "cell_acc": 0.0,
            "nonzero_iou": 0.0,
            "color_hist_l1": 0.0,
            "mean_token_log_prob": 0.0,
            "mean_confidence": 0.0,
        }
    else:
        summary = {
            "exact": exact / total,
            "valid": 1.0,
            "cell_acc": correct_cells / max(total_cells, 1),
            "nonzero_iou": nonzero_iou_sum / total,
            "color_hist_l1": color_hist_l1_sum / total,
            "mean_token_log_prob": token_log_prob_sum / max(total_cells, 1),
            "mean_confidence": confidence_sum / max(total_cells, 1),
        }
    if eval_report is not None:
        _write_json(
            eval_report,
            {
                "summary": summary,
                "examples": examples,
                "metadata": {
                    "architecture": "structured_encoder",
                    "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
                    "inference_mode": inference_mode,
                    "num_candidates": num_candidates,
                    "sample_steps": sample_steps,
                    "temperature_start": temperature_start,
                    "temperature_end": temperature_end,
                    "ensemble_strategy": ensemble_strategy,
                    "args": _jsonable_args(args),
                },
            },
        )
    return summary


def build_serialized_dataset(args: argparse.Namespace) -> tuple[ArcDataset, ArcDataset | None]:
    train_files = find_arc_json_files(args.train_dir or args.data_dir)
    if not train_files:
        raise FileNotFoundError("no ARC train JSON files found")
    train_dataset = ArcDataset(train_files, seq_len=args.seq_len)

    eval_dataset = None
    eval_files = find_arc_json_files(args.eval_dir)
    if eval_files:
        eval_dataset = ArcDataset(eval_files, seq_len=args.seq_len)
    return train_dataset, eval_dataset


def build_structured_dataset(
    args: argparse.Namespace,
) -> tuple[ArcStructuredDataset | None, ArcStructuredDataset | None]:
    train_files = [] if args.eval_only else find_arc_json_files(args.train_dir or args.data_dir)
    if not train_files:
        if args.eval_only:
            train_dataset = None
        else:
            raise FileNotFoundError("no ARC train JSON files found")
    else:
        train_dataset = ArcStructuredDataset(
            train_files,
            max_grid_size=args.max_grid_size,
            max_examples=args.max_examples,
            augment_color_permutation=args.augment_color_permutation,
            augment_translation=args.augment_translation,
            augment_grid_noise=args.augment_grid_noise,
            curriculum=args.curriculum,
        )

    eval_dataset = None
    eval_files = find_arc_json_files(args.eval_dir)
    if eval_files:
        eval_dataset = ArcStructuredDataset(
            eval_files,
            max_grid_size=args.max_grid_size,
            max_examples=args.max_examples,
        )
    return train_dataset, eval_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing ARC task JSON",
    )
    parser.add_argument("--train-dir", type=Path, default=None, help="Training ARC JSON directory")
    parser.add_argument("--eval-dir", type=Path, default=None, help="Evaluation ARC JSON directory")
    parser.add_argument(
        "--arch",
        choices=["serialized", "structured_encoder"],
        default="structured_encoder",
        help="ARC architecture to train",
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-grid-size", type=int, default=30)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Checkpoint structured refinement blocks to keep full gradients with lower memory.",
    )
    parser.add_argument(
        "--stochastic-depth-prob",
        type=float,
        default=0.0,
        help="Drop non-final structured refinement blocks with this probability during training.",
    )
    parser.add_argument(
        "--aux-loss-weight",
        type=float,
        default=0.0,
        help="Weight for intermediate structured refinement losses.",
    )
    parser.add_argument(
        "--use-object-features",
        action="store_true",
        help="Condition the structured encoder on connected-component object metadata.",
    )
    parser.add_argument(
        "--augment-color-permutation",
        action="store_true",
        help="Add a consistent color-permuted copy of each structured ARC task.",
    )
    parser.add_argument(
        "--augment-translation",
        action="store_true",
        help="Add a translated copy of each structured ARC task when nonzero cells fit in bounds.",
    )
    parser.add_argument(
        "--augment-grid-noise",
        action="store_true",
        help="Add a copy of each structured ARC task with sparse input-only distractor cells.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Sort structured ARC examples from smaller/simpler to larger/harder.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-limit", type=int, default=50)
    parser.add_argument("--sample-steps", type=int, default=64)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-report", type=Path, default=None)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--debug-limit", type=int, default=5)
    parser.add_argument(
        "--inference-mode",
        choices=["greedy", "ensemble"],
        default="greedy",
        help="Inference mode for structured eval.",
    )
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.1)
    parser.add_argument(
        "--ensemble-strategy",
        choices=["confidence", "majority"],
        default="confidence",
    )
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/arc"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def move_tensor_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def train_serialized(args: argparse.Namespace, device: str) -> None:
    train_dataset, eval_dataset = build_serialized_dataset(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_arc_examples,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )
    train_iter = iter(train_loader)
    model = create_model(dim=args.dim, seq_len=args.seq_len).to(device)
    run_training_loop(
        args,
        device,
        model,
        train_dataset,
        eval_dataset,
        train_loader,
        train_iter,
        evaluate,
    )


def train_structured(args: argparse.Namespace, device: str) -> None:
    train_dataset, eval_dataset = build_structured_dataset(args)
    model = create_structured_model(args).to(device)
    if args.eval_only:
        if eval_dataset is None:
            raise SystemExit("provide --eval-dir with ARC JSON files for --eval-only")
        if args.resume is None:
            raise SystemExit("provide --resume for --eval-only")
        load_model_checkpoint(args.resume, model, device)
        metrics = evaluate_structured(
            model,
            eval_dataset,
            device,
            args.eval_limit,
            args.sample_steps,
            inference_mode=args.inference_mode,
            num_candidates=args.num_candidates,
            temperature_start=args.temperature_start,
            temperature_end=args.temperature_end,
            ensemble_strategy=args.ensemble_strategy,
            eval_report=args.eval_report,
            debug_dir=args.debug_dir,
            debug_limit=args.debug_limit,
            args=args,
            checkpoint_path=args.resume,
        )
        print(
            f"eval exact={metrics['exact']:.3%} valid={metrics['valid']:.3%} "
            f"cell_acc={metrics['cell_acc']:.3%}"
        )
        return
    if train_dataset is None:
        raise FileNotFoundError("no ARC train JSON files found")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_structured_arc_examples,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )
    train_iter = iter(train_loader)
    run_training_loop(
        args,
        device,
        model,
        train_dataset,
        eval_dataset,
        train_loader,
        train_iter,
        evaluate_structured,
    )


def run_training_loop(
    args: argparse.Namespace,
    device: str,
    model: ArchitectureModel,
    train_dataset: ArcDataset | ArcStructuredDataset,
    eval_dataset: ArcDataset | ArcStructuredDataset | None,
    train_loader: DataLoader,
    train_iter,
    eval_fn,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer, scheduler, device)

    total_params = sum(p.numel() for p in model.parameters())
    eval_count = len(eval_dataset) if eval_dataset else 0
    print(f"Architecture: {args.arch}")
    print(f"ARC examples: train={len(train_dataset)}, eval={eval_count}")
    print(f"Model: {total_params:,} params on {device}")
    print(f"{'Step':>8} {'Loss':>10} {'Acc%':>8} {'Mask%':>8} {'LR':>10} {'Time':>8}")
    print("-" * 60)

    start_time = time.time()
    model.train()
    last_out: dict[str, Any] | None = None
    for step in range(start_step, args.steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        moved = move_tensor_batch(batch, device)
        optimizer.zero_grad()
        if args.arch == "serialized":
            out = model(
                moved["tokens"],
                padding_mask=moved["padding_mask"],
                loss_mask=moved["loss_mask"],
            )
        else:
            out = model(**structured_forward_kwargs(moved))
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        last_out = out

        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.time() - start_time
            print(
                f"{step:>8} {out['loss'].item():>10.4f} "
                f"{out['masked_acc'].item() * 100:>7.1f}% "
                f"{out['masked_ratio'].item() * 100:>7.1f}% "
                f"{optimizer.param_groups[0]['lr']:>10.2e} {elapsed:>7.1f}s"
            )

        if eval_dataset is not None and step > 0 and step % args.eval_every == 0:
            if args.arch == "structured_encoder":
                metrics = eval_fn(
                    model,
                    eval_dataset,
                    device,
                    args.eval_limit,
                    args.sample_steps,
                    inference_mode=args.inference_mode,
                    num_candidates=args.num_candidates,
                    temperature_start=args.temperature_start,
                    temperature_end=args.temperature_end,
                    ensemble_strategy=args.ensemble_strategy,
                    eval_report=args.eval_report,
                    debug_dir=args.debug_dir,
                    debug_limit=args.debug_limit,
                    args=args,
                    checkpoint_path=args.resume,
                )
            else:
                metrics = eval_fn(model, eval_dataset, device, args.eval_limit, args.sample_steps)
            if args.arch == "structured_encoder":
                print(
                    f"eval exact={metrics['exact']:.3%} valid={metrics['valid']:.3%} "
                    f"cell_acc={metrics['cell_acc']:.3%}"
                )
            else:
                print(f"eval exact={metrics['exact']:.3%} valid={metrics['valid']:.3%}")

        if step > 0 and step % args.save_every == 0:
            save_checkpoint(
                args.checkpoint_dir / f"step_{step}.pt",
                model,
                optimizer,
                scheduler,
                step,
                args,
            )
            save_checkpoint(
                args.checkpoint_dir / "latest.pt",
                model,
                optimizer,
                scheduler,
                step,
                args,
            )

    if last_out is not None:
        save_checkpoint(
            args.checkpoint_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            args.steps - 1,
            args,
        )
        print(f"Training complete. Final loss: {last_out['loss'].item():.4f}")


def main() -> None:
    args = parse_args()
    if not args.eval_only and args.data_dir is None and args.train_dir is None:
        raise SystemExit("provide --data-dir or --train-dir")
    if args.eval_only and args.arch != "structured_encoder":
        raise SystemExit("--eval-only is currently supported only for --arch structured_encoder")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        device = choose_device(args.device)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.arch == "serialized":
        train_serialized(args, device)
    else:
        train_structured(args, device)


if __name__ == "__main__":
    main()
