"""
train.py — Minimal training loop for Tiny Recursive Model.

Stripped down to debug the core learning dynamics.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from rdlm.trm import TinyRecursiveModel


class TinyTextDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 16):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.seq_len = seq_len

        self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)

        # Sliding window: predict next token
        self.examples = []
        for i in range(len(self.data) - seq_len - 1):
            x = self.data[i : i + seq_len]
            y = self.data[i + 1 : i + seq_len + 1]
            self.examples.append((x, y))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def compute_accuracy(logits, labels):
    """Accuracy of argmax predictions."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def main():
    text = """
    The cat sat on the mat and the dog sat on the log.
    The bird sat on the tree and the fish swam in the sea.
    The cat chased the mouse and the dog chased the cat.
    The bird flew over the house and the fish swam under the bridge.
    """.strip()

    seq_len = 16
    dataset = TinyTextDataset(text, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Vocab: {dataset.vocab_size}, Examples: {len(dataset)}, Seq len: {seq_len}")
    print(f"Chars: {''.join(dataset.idx_to_char[i] for i in range(dataset.vocab_size))}")
    print()

    # Model (no frills)
    model = TinyRecursiveModel(
        vocab_size=dataset.vocab_size,
        dim=128,
        num_heads=4,
        max_seq_len=seq_len + 32,
        num_latent_refinements=4,
        num_refinement_blocks=2,
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params, running on {device}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)

    model.train()
    losses = []
    accuracies = []
    print(f"{'Step':>6} {'Loss':>10} {'Acc':>8} {'LR':>10} {'Alphas':>14}")
    print("-" * 52)

    for step in range(200):
        tokens, labels = next(iter(dataloader))
        tokens, labels = tokens.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(tokens, labels=labels)
        loss = out["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        acc = compute_accuracy(out["logits"], labels)
        losses.append(out["token_loss"].item())
        accuracies.append(acc)

        if step % 20 == 0:
            alphas = f"{model.network.alpha_attn.item():.4f}/{model.network.alpha_ff.item():.4f}"
            print(
                f"{step:>6} {out['token_loss'].item():>10.4f} "
                f"{acc:>7.3%} {optimizer.param_groups[0]['lr']:>10.2e} "
                f"{alphas:>14}"
            )

        if out["token_loss"].item() < 0.5:
            print(f"\n✅ Model learned well! Loss below 0.5 at step {step}")
            break

    print(f"\nFinal loss: {losses[-1]:.4f}, accuracy: {accuracies[-1]:.3%}")

    # Generate: check if it learned the pattern
    model.eval()
    with torch.no_grad():
        for prompt_text in ["The cat", "The dog", "The bird"]:
            indices = [dataset.char_to_idx.get(c, 0) for c in prompt_text]
            prompt = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)
            generated = model.generate_autoregressive(prompt, max_new_tokens=20)
            decoded = "".join(dataset.idx_to_char.get(i.item(), "?") for i in generated[0])
            print(f'  "{prompt_text}" → "{decoded}"')


if __name__ == "__main__":
    main()
