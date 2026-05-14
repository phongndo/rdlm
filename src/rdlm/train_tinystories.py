"""
train_tinystories.py — Train Recursive Diffusion LM on TinyStories benchmark.

Trains a tiny recursive diffusion language model on the TinyStories dataset,
a standard benchmark for small language models.

Usage:
    uv run python3 src/rdlm/train_tinystories.py --steps 20000 --dim 256

Metrics tracked:
    - Training loss / perplexity
    - Validation loss / perplexity
    - Generation samples
"""

import argparse
import math
import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from rdlm.diffusion_lm import NoiseSchedule, RecursiveDiffusionLM
from rdlm.trm import TinyRecursiveModel


class TinyStoriesCharDataset(IterableDataset):
    """Character-level TinyStories dataset for diffusion LM training."""

    def __init__(self, split: str = "train", seq_len: int = 128, max_examples: int | None = None):
        self.split = split
        self.seq_len = seq_len
        self.max_examples = max_examples

        # Build character vocabulary from a sample
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        chars = set()
        for i, example in enumerate(ds):
            chars.update(example["text"])
            if i >= 1000:
                break
        del ds  # discard

        chars = sorted(chars)
        self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        # Reserve mask token
        self.mask_token_id = self.vocab_size
        self.vocab_size_with_mask = self.vocab_size + 1

    def __iter__(self):
        # Re-load to get fresh iterator
        ds = load_dataset("roneneldan/TinyStories", split=self.split, streaming=True)
        count = 0
        for example in ds:
            text = example["text"]
            # Skip very short stories
            if len(text) < self.seq_len + 1:
                continue

            # Extract sliding window chunks (50% overlap)
            encoded = [self.char_to_idx.get(c, 0) for c in text]
            for i in range(0, len(encoded) - self.seq_len, self.seq_len // 2):
                chunk = encoded[i : i + self.seq_len]
                yield torch.tensor(chunk, dtype=torch.long)

                count += 1
                if self.max_examples and count >= self.max_examples:
                    return

    def __len__(self):
        return self.max_examples or 100000


def create_model(
    vocab_size_with_mask: int,
    dim: int = 128,
    seq_len: int = 128,
) -> RecursiveDiffusionLM:
    """Create the recursive diffusion LM."""
    backbone = TinyRecursiveModel(
        vocab_size=vocab_size_with_mask,
        dim=dim,
        num_heads=4,
        max_seq_len=seq_len * 2,
        num_latent_refinements=6,
        num_refinement_blocks=2,
    )

    diff_lm = RecursiveDiffusionLM(
        backbone=backbone,
        mask_token_id=vocab_size_with_mask - 1,  # last token is mask
        noise_schedule=NoiseSchedule(),
    )

    return diff_lm


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return math.exp(loss)


def generate_samples(
    diff_lm: RecursiveDiffusionLM,
    dataset: TinyStoriesCharDataset,
    device: str,
    num_samples: int = 3,
    prompt_texts: list[str] | None = None,
):
    """Generate sample stories."""
    diff_lm.eval()

    if prompt_texts is None:
        prompt_texts = ["Once upon a time", "She", "The cat"]

    print(f"\n{'=' * 60}")
    print("Generation Samples")
    print(f"{'=' * 60}")

    for prompt_text in prompt_texts:
        # Encode prompt
        indices = []
        for c in prompt_text:
            if c in dataset.char_to_idx:
                indices.append(dataset.char_to_idx[c])
            else:
                indices.append(0)  # fallback

        prompt = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            max_new = dataset.seq_len - len(indices)
            if max_new < 8:
                max_new = dataset.seq_len

            generated = diff_lm.sample(
                prompt=prompt,
                max_new_tokens=min(max_new, dataset.seq_len),
                steps=64,
                block_size=dataset.seq_len,
            )

            text = "".join(dataset.idx_to_char.get(t.item(), "�") for t in generated[0])
            print(f"\n  Prompt: {prompt_text!r}")
            print(f"  Story:  {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=128, help="Model dimension")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--log-every", type=int, default=500, help="Log every N steps")
    parser.add_argument("--eval-samples", type=int, default=5, help="Generate samples every N logs")
    args = parser.parse_args()

    # ── Dataset ──
    print("Loading TinyStories dataset...")
    train_dataset = TinyStoriesCharDataset(split="train", seq_len=args.seq_len, max_examples=50000)
    print(f"Character vocab: {train_dataset.vocab_size} chars (+1 [MASK])")
    chars = "".join(train_dataset.idx_to_char[i] for i in range(train_dataset.vocab_size))
    print(f"Characters: {chars}")
    print()

    # ── Model ──
    print(f"Creating model (dim={args.dim}, seq_len={args.seq_len})...")
    diff_lm = create_model(train_dataset.vocab_size_with_mask, dim=args.dim, seq_len=args.seq_len)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    diff_lm = diff_lm.to(device)

    total_params = sum(p.numel() for p in diff_lm.parameters())
    print(f"Model: {total_params:,} params on {device}")
    print(f"  - TinyBlock: {sum(p.numel() for p in diff_lm.backbone.network.parameters()):,}")
    effective_depth = (
        diff_lm.backbone.num_latent_refinements * diff_lm.backbone.num_refinement_blocks
    )
    print(
        f"  - Recursion: {diff_lm.backbone.num_latent_refinements}x "
        f"x {diff_lm.backbone.num_refinement_blocks}x = {effective_depth} effective depth"
    )
    print()

    # ── Training ──
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(diff_lm.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    diff_lm.train()

    print(f"{'Step':>8} {'Loss':>10} {'PPL':>10} {'Acc%':>8} {'LR':>10} {'Time':>8}")
    print("-" * 56)

    start_time = time.time()
    train_iter = iter(train_loader)

    for step in range(args.steps):
        # Get batch
        try:
            x0 = next(train_iter).to(device)
        except StopIteration:
            train_iter = iter(train_loader)
            x0 = next(train_iter).to(device)

        # Forward + backward
        optimizer.zero_grad()
        out = diff_lm(x0)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(diff_lm.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.time() - start_time
            ppl = compute_perplexity(out["loss"].item())
            acc_pct = out["masked_acc"].item() * 100
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"{step:>8} {out['loss'].item():>10.4f} "
                f"{ppl:>10.2f} {acc_pct:>7.1f}% "
                f"{lr:>10.2e} {elapsed:>7.1f}s"
            )

            # Generate samples every few logs
            should_sample = step > 0 and (
                step % (args.log_every * args.eval_samples) == 0 or step == args.steps - 1
            )
            if should_sample:
                generate_samples(diff_lm, train_dataset, device)

    print()
    print("=" * 60)
    print("Training Complete!")
    final_loss = out["loss"].item()
    print(f"Final loss: {final_loss:.4f}, perplexity: {compute_perplexity(final_loss):.2f}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print("=" * 60)

    # Final generation
    generate_samples(
        diff_lm,
        train_dataset,
        device,
        prompt_texts=[
            "Once upon a time",
            "She",
            "The cat",
            "He was",
            "In the forest",
        ],
    )


if __name__ == "__main__":
    main()
