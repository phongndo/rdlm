"""ARC-AGI JSON loading and serialization utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

EOS = "~"
PAD = "<PAD>"
MASK = "<MASK>"
ARC_CHARS = sorted(set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_:\n/ " + EOS))
TOKENS = [PAD, *ARC_CHARS, MASK]
PAD_TOKEN_ID = 0
MASK_TOKEN_ID = len(TOKENS) - 1
CHAR_TO_ID = {token: idx for idx, token in enumerate(TOKENS)}
ID_TO_CHAR = {idx: token for token, idx in CHAR_TO_ID.items()}
VOCAB_SIZE = len(TOKENS)


Grid = list[list[int]]
Pair = dict[str, Grid]


@dataclass(frozen=True)
class ArcExample:
    """A serialized ARC training example."""

    task_id: str
    prompt: str
    target: str
    tokens: tuple[int, ...]
    loss_mask: tuple[bool, ...]


def find_arc_json_files(*roots: str | Path | None) -> list[Path]:
    """Find ARC task JSON files below the provided roots."""
    files: list[Path] = []
    for root in roots:
        if root is None:
            continue
        path = Path(root)
        if path.is_file() and path.suffix == ".json":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
    return sorted(set(files))


def load_arc_task(path: str | Path) -> dict[str, Any]:
    """Load one ARC task JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        task = json.load(f)
    if not isinstance(task.get("train"), list) or not isinstance(task.get("test"), list):
        raise ValueError(f"{path} is not an ARC task JSON file")
    return task


def serialize_grid(grid: Grid) -> str:
    """Serialize an ARC grid as row strings separated by '/'."""
    if not grid or not all(isinstance(row, list) and row for row in grid):
        raise ValueError("ARC grids must be non-empty rectangular lists")
    width = len(grid[0])
    rows: list[str] = []
    for row in grid:
        if len(row) != width:
            raise ValueError("ARC grids must be rectangular")
        if not all(isinstance(cell, int) and 0 <= cell <= 9 for cell in row):
            raise ValueError("ARC grid cells must be integers from 0 to 9")
        rows.append("".join(str(cell) for cell in row))
    return "/".join(rows)


def parse_grid(text: str) -> Grid | None:
    """Parse a serialized ARC grid, returning None on malformed output."""
    text = text.split(EOS, 1)[0].strip()
    if not text:
        return None
    rows = text.split("/")
    width = len(rows[0])
    if width == 0:
        return None
    grid: Grid = []
    for row in rows:
        if len(row) != width or any(ch not in "0123456789" for ch in row):
            return None
        grid.append([int(ch) for ch in row])
    return grid


def encode_text(text: str) -> tuple[int, ...]:
    """Encode text using the fixed ARC serialization vocabulary."""
    try:
        return tuple(CHAR_TO_ID[ch] for ch in text)
    except KeyError as exc:
        raise ValueError(f"character {exc.args[0]!r} is not in the ARC vocabulary") from exc


def decode_tokens(tokens: list[int] | torch.Tensor) -> str:
    """Decode token IDs, ignoring PAD and stopping before MASK tokens."""
    if isinstance(tokens, torch.Tensor):
        token_ids = [int(t) for t in tokens.tolist()]
    else:
        token_ids = [int(t) for t in tokens]
    chars: list[str] = []
    for token_id in token_ids:
        if token_id == PAD_TOKEN_ID:
            continue
        if token_id == MASK_TOKEN_ID:
            break
        chars.append(ID_TO_CHAR.get(token_id, ""))
    return "".join(chars)


def _pair_block(pair: Pair) -> str:
    return f"IN\n{serialize_grid(pair['input'])}\nOUT\n{serialize_grid(pair['output'])}\n"


def serialize_arc_example(demos: list[Pair], query_input: Grid, output: Grid) -> tuple[str, str]:
    """Serialize demos plus query input into a prompt and answer target."""
    prompt_parts = ["TASK\n"]
    for demo in demos:
        prompt_parts.append("DEMO\n")
        prompt_parts.append(_pair_block(demo))
    prompt_parts.append(f"QUERY\nIN\n{serialize_grid(query_input)}\nOUT\n")
    target = f"{serialize_grid(output)}{EOS}"
    return "".join(prompt_parts), target


def build_examples_from_task(task_id: str, task: dict[str, Any], seq_len: int) -> list[ArcExample]:
    """Build serialized training examples from one ARC task."""
    train_pairs: list[Pair] = [
        pair for pair in task["train"] if "input" in pair and "output" in pair
    ]
    test_pairs: list[Pair] = [
        pair for pair in task["test"] if "input" in pair and "output" in pair
    ]
    raw_examples: list[tuple[list[Pair], Grid, Grid]] = []

    for idx, pair in enumerate(train_pairs):
        demos = [demo for demo_idx, demo in enumerate(train_pairs) if demo_idx != idx]
        raw_examples.append((demos, pair["input"], pair["output"]))
    for pair in test_pairs:
        raw_examples.append((train_pairs, pair["input"], pair["output"]))

    examples: list[ArcExample] = []
    for demos, query_input, output in raw_examples:
        prompt, target = serialize_arc_example(demos, query_input, output)
        prompt_ids = encode_text(prompt)
        target_ids = encode_text(target)
        token_ids = prompt_ids + target_ids
        if len(token_ids) > seq_len:
            continue
        examples.append(
            ArcExample(
                task_id=task_id,
                prompt=prompt,
                target=target,
                tokens=token_ids,
                loss_mask=(False,) * len(prompt_ids) + (True,) * len(target_ids),
            )
        )
    return examples


class ArcDataset(Dataset):
    """Map-style dataset of serialized ARC examples."""

    def __init__(self, files: list[str | Path], seq_len: int):
        self.seq_len = seq_len
        self.examples: list[ArcExample] = []
        self.skipped_tasks = 0
        for path in files:
            try:
                task = load_arc_task(path)
                examples = build_examples_from_task(Path(path).stem, task, seq_len=seq_len)
            except ValueError:
                self.skipped_tasks += 1
                continue
            self.examples.extend(examples)
        if not self.examples:
            raise ValueError("no ARC examples fit within seq_len")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ArcExample:
        return self.examples[idx]


def collate_arc_examples(batch: list[ArcExample]) -> dict[str, torch.Tensor | list[str]]:
    """Pad ARC examples into tensors for diffusion training."""
    max_len = max(len(example.tokens) for example in batch)
    tokens = torch.full((len(batch), max_len), PAD_TOKEN_ID, dtype=torch.long)
    padding_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    loss_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    task_ids: list[str] = []
    for row, example in enumerate(batch):
        length = len(example.tokens)
        tokens[row, :length] = torch.tensor(example.tokens, dtype=torch.long)
        padding_mask[row, :length] = True
        loss_mask[row, :length] = torch.tensor(example.loss_mask, dtype=torch.bool)
        task_ids.append(example.task_id)
    return {
        "tokens": tokens,
        "padding_mask": padding_mask,
        "loss_mask": loss_mask,
        "task_ids": task_ids,
    }
