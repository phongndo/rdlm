"""
train_diffusion.py — Train the Recursive Diffusion Language Model.

Trains using masked diffusion (MDLM) loss and demonstrates generation.
"""

import time

import torch
from torch.utils.data import DataLoader, Dataset

from rdlm.diffusion_lm import NoiseSchedule, RecursiveDiffusionLM
from rdlm.trm import TinyRecursiveModel


class CharLevelDataset(Dataset):
    """Simple character-level text dataset for diffusion LM training.

    Unlike autoregressive training (predict next token),
    diffusion training predicts masked tokens anywhere in the sequence.
    """

    def __init__(self, text: str, seq_len: int = 32):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.seq_len = seq_len

        # We reserve the last token ID for [MASK]
        self.mask_token_id = self.vocab_size
        self.vocab_size_with_mask = self.vocab_size + 1

        # Encode text
        encoded = [self.char_to_idx[c] for c in text]

        # Create fixed-length chunks
        self.chunks = []
        for i in range(0, len(encoded), seq_len):
            chunk = encoded[i : i + seq_len]
            if len(chunk) == seq_len:
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))
            elif len(chunk) >= seq_len // 2:
                # Pad short last chunk
                padded = chunk + [0] * (seq_len - len(chunk))
                self.chunks.append(torch.tensor(padded, dtype=torch.long))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def decode_tokens(tokens, idx_to_char, mask_token_id=None):
    """Decode token IDs to string, replacing [MASK] with '▮'."""
    result = []
    for t in tokens:
        if mask_token_id is not None and t == mask_token_id:
            result.append("▮")
        else:
            result.append(idx_to_char.get(t, "?"))
    return "".join(result)


def main():
    text = """\
The cat sat on the mat and the dog sat on the log.
The bird sat on the tree and the fish swam in the sea.
The cat chased the mouse and the dog chased the cat.
The bird flew over the house and the fish swam under the bridge.
The sun rose in the east and set in the west each day.
The stars twinkled in the night sky bright and clear.
The rain fell on the roof and dripped down to the ground.
The wind blew through the trees and shook the leaves around.
The fish swam in the pond and the ducks quacked on the bank.
The cow stood in the field and the horse ran in the park."""

    seq_len = 32
    dataset = CharLevelDataset(text, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Vocab: {dataset.vocab_size} chars (+1 for [MASK])")
    print(f"Data: {len(dataset)} examples of length {seq_len}")
    print(f"Chars: {''.join(dataset.idx_to_char[i] for i in range(dataset.vocab_size))}")
    print()

    # Create backbone
    backbone = TinyRecursiveModel(
        vocab_size=dataset.vocab_size_with_mask,
        dim=128,
        num_heads=4,
        max_seq_len=seq_len * 2,
        num_latent_refinements=6,
        num_refinement_blocks=2,
    )

    # Create diffusion LM
    diff_lm = RecursiveDiffusionLM(
        backbone=backbone,
        mask_token_id=dataset.mask_token_id,
        noise_schedule=NoiseSchedule(),
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    diff_lm = diff_lm.to(device)
    total_params = sum(p.numel() for p in diff_lm.parameters())
    print(f"Model: {total_params:,} params on {device}")
    print()

    optimizer = torch.optim.AdamW(diff_lm.parameters(), lr=3e-4, weight_decay=0.0)
    diff_lm.train()

    print(f"{'Step':>6} {'Loss':>10} {'Mask%':>8} {'Acc%':>8} {'Time':>8}")
    print("-" * 44)

    start_time = time.time()
    for step in range(2000):
        x0 = next(iter(dataloader)).to(device)

        optimizer.zero_grad()
        out = diff_lm(x0)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(diff_lm.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0:
            elapsed = time.time() - start_time
            masked_pct = out["masked_ratio"].item() * 100
            acc_pct = out["masked_acc"].item() * 100
            print(
                f"{step:>6} {out['loss'].item():>10.4f} "
                f"{masked_pct:>7.1f}% {acc_pct:>7.1f}% "
                f"{elapsed:>7.1f}s"
            )

    print()
    print("=" * 60)
    print("Generation: Iterative Unmasking")
    print("=" * 60)

    diff_lm.eval()

    # Generate from scratch (no prompt)
    print("\n--- From scratch (no prompt) ---")
    with torch.no_grad():
        generated = diff_lm.sample(
            prompt=None,
            max_new_tokens=seq_len,
            steps=64,
            block_size=seq_len,
        )
        text = decode_tokens(generated[0].tolist(), dataset.idx_to_char, dataset.mask_token_id)
        print(f"  {text!r}")

    # Generate with prompt
    print("\n--- With prompt ---")
    for prompt_text in ["The cat", "The dog", "The bird"]:
        indices = [dataset.char_to_idx[c] for c in prompt_text]
        prompt = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            generated = diff_lm.sample(
                prompt=prompt,
                max_new_tokens=seq_len - len(prompt_text),
                steps=64,
                block_size=seq_len,
            )
            text = decode_tokens(generated[0], dataset.idx_to_char, dataset.mask_token_id)
            print(f"  Prompt: {prompt_text!r}")
            print(f"  Result: {text!r}")
            print()


if __name__ == "__main__":
    main()
