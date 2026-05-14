"""Train the recursive diffusion LM on serialized ARC-AGI JSON tasks."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

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
from rdlm.arc_model import ArcOutputDiffusion
from rdlm.diffusion_lm import NoiseSchedule, RecursiveDiffusionLM
from rdlm.trm import TinyRecursiveModel

ArchitectureModel = RecursiveDiffusionLM | ArcOutputDiffusion


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
) -> dict[str, float]:
    model.eval()
    total = min(limit, len(dataset))
    exact = 0

    for idx in range(total):
        batch = collate_structured_arc_examples([dataset[idx]])
        moved = {
            key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)
        }
        generated = model.sample(
            context_colors=moved["context_colors"],
            context_rows=moved["context_rows"],
            context_cols=moved["context_cols"],
            context_roles=moved["context_roles"],
            context_examples=moved["context_examples"],
            context_mask=moved["context_mask"],
            target_rows=moved["target_rows"],
            target_cols=moved["target_cols"],
            target_mask=moved["target_mask"],
            steps=sample_steps,
        )
        expected = moved["target_colors"]
        target_mask = moved["target_mask"].bool()
        if torch.equal(generated[target_mask].cpu(), expected[target_mask].cpu()):
            exact += 1

    model.train()
    if total == 0:
        return {"exact": 0.0, "valid": 1.0}
    return {"exact": exact / total, "valid": 1.0}


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
) -> tuple[ArcStructuredDataset, ArcStructuredDataset | None]:
    train_files = find_arc_json_files(args.train_dir or args.data_dir)
    if not train_files:
        raise FileNotFoundError("no ARC train JSON files found")
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_structured_arc_examples,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )
    train_iter = iter(train_loader)
    model = create_structured_model(args).to(device)
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
            out = model(
                context_colors=moved["context_colors"],
                context_rows=moved["context_rows"],
                context_cols=moved["context_cols"],
                context_roles=moved["context_roles"],
                context_examples=moved["context_examples"],
                context_mask=moved["context_mask"],
                target_colors=moved["target_colors"],
                target_rows=moved["target_rows"],
                target_cols=moved["target_cols"],
                target_mask=moved["target_mask"],
            )
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
            metrics = eval_fn(model, eval_dataset, device, args.eval_limit, args.sample_steps)
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
    if args.data_dir is None and args.train_dir is None:
        raise SystemExit("provide --data-dir or --train-dir")

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
