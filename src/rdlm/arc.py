"""ARC-AGI JSON loading and serialization utilities."""

from __future__ import annotations

import json
import random
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
ROLE_DEMO_INPUT = 0
ROLE_DEMO_OUTPUT = 1
ROLE_QUERY_INPUT = 2
ROLE_TARGET_OUTPUT = 3


@dataclass(frozen=True)
class ArcExample:
    """A serialized ARC training example."""

    task_id: str
    prompt: str
    target: str
    tokens: tuple[int, ...]
    loss_mask: tuple[bool, ...]


@dataclass(frozen=True)
class GridObjectFeatures:
    """Per-cell object metadata for one ARC grid."""

    object_ids: Grid
    size_buckets: Grid
    heights: Grid
    widths: Grid
    rel_rows: Grid
    rel_cols: Grid


@dataclass(frozen=True)
class ArcStructuredExample:
    """Grid-native ARC example for encoder-conditioned output diffusion."""

    task_id: str
    context_colors: tuple[int, ...]
    context_rows: tuple[int, ...]
    context_cols: tuple[int, ...]
    context_roles: tuple[int, ...]
    context_examples: tuple[int, ...]
    context_object_ids: tuple[int, ...]
    context_object_size_buckets: tuple[int, ...]
    context_object_heights: tuple[int, ...]
    context_object_widths: tuple[int, ...]
    context_object_rel_rows: tuple[int, ...]
    context_object_rel_cols: tuple[int, ...]
    target_colors: tuple[int, ...]
    target_rows: tuple[int, ...]
    target_cols: tuple[int, ...]
    target_height: int
    target_width: int


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
    test_pairs: list[Pair] = [pair for pair in task["test"] if "input" in pair and "output" in pair]
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


def _zero_feature_grid(grid: Grid) -> Grid:
    return [[0 for _cell in row] for row in grid]


def compute_grid_object_features(grid: Grid) -> GridObjectFeatures:
    """Compute deterministic 4-connected same-color object features.

    Non-zero cells are grouped into objects. Background cells keep zero-valued
    object features so they can share one embedding bucket.
    """
    serialize_grid(grid)
    height = len(grid)
    width = len(grid[0])
    visited = [[False for _cell in row] for row in grid]
    object_ids = _zero_feature_grid(grid)
    size_buckets = _zero_feature_grid(grid)
    heights = _zero_feature_grid(grid)
    widths = _zero_feature_grid(grid)
    rel_rows = _zero_feature_grid(grid)
    rel_cols = _zero_feature_grid(grid)
    next_object_id = 1

    for start_row, row in enumerate(grid):
        for start_col, color in enumerate(row):
            if color == 0 or visited[start_row][start_col]:
                continue

            queue = [(start_row, start_col)]
            visited[start_row][start_col] = True
            cells: list[tuple[int, int]] = []
            for row_idx, col_idx in queue:
                cells.append((row_idx, col_idx))
                for next_row, next_col in (
                    (row_idx - 1, col_idx),
                    (row_idx + 1, col_idx),
                    (row_idx, col_idx - 1),
                    (row_idx, col_idx + 1),
                ):
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if visited[next_row][next_col] or grid[next_row][next_col] != color:
                        continue
                    visited[next_row][next_col] = True
                    queue.append((next_row, next_col))

            rows = [row_idx for row_idx, _col_idx in cells]
            cols = [col_idx for _row_idx, col_idx in cells]
            min_row = min(rows)
            min_col = min(cols)
            object_height = max(rows) - min_row + 1
            object_width = max(cols) - min_col + 1
            size_bucket = len(cells).bit_length()
            for row_idx, col_idx in cells:
                object_ids[row_idx][col_idx] = next_object_id
                size_buckets[row_idx][col_idx] = size_bucket
                heights[row_idx][col_idx] = object_height
                widths[row_idx][col_idx] = object_width
                rel_rows[row_idx][col_idx] = row_idx - min_row + 1
                rel_cols[row_idx][col_idx] = col_idx - min_col + 1
            next_object_id += 1

    return GridObjectFeatures(
        object_ids=object_ids,
        size_buckets=size_buckets,
        heights=heights,
        widths=widths,
        rel_rows=rel_rows,
        rel_cols=rel_cols,
    )


def _append_grid_cells(
    grid: Grid,
    role: int,
    example_idx: int,
    colors: list[int],
    rows: list[int],
    cols: list[int],
    roles: list[int],
    examples: list[int],
    object_ids: list[int],
    object_size_buckets: list[int],
    object_heights: list[int],
    object_widths: list[int],
    object_rel_rows: list[int],
    object_rel_cols: list[int],
) -> None:
    serialize_grid(grid)
    object_features = compute_grid_object_features(grid)
    for row_idx, row in enumerate(grid):
        for col_idx, color in enumerate(row):
            colors.append(color)
            rows.append(row_idx)
            cols.append(col_idx)
            roles.append(role)
            examples.append(example_idx)
            object_ids.append(object_features.object_ids[row_idx][col_idx])
            object_size_buckets.append(object_features.size_buckets[row_idx][col_idx])
            object_heights.append(object_features.heights[row_idx][col_idx])
            object_widths.append(object_features.widths[row_idx][col_idx])
            object_rel_rows.append(object_features.rel_rows[row_idx][col_idx])
            object_rel_cols.append(object_features.rel_cols[row_idx][col_idx])


def _grid_fits(grid: Grid, max_grid_size: int) -> bool:
    return bool(grid) and len(grid) <= max_grid_size and len(grid[0]) <= max_grid_size


def _copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def _copy_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "train": [
            {"input": _copy_grid(pair["input"]), "output": _copy_grid(pair["output"])}
            for pair in task["train"]
            if "input" in pair and "output" in pair
        ],
        "test": [
            {"input": _copy_grid(pair["input"]), "output": _copy_grid(pair["output"])}
            for pair in task["test"]
            if "input" in pair and "output" in pair
        ],
    }


def _permute_grid_colors(grid: Grid, permutation: list[int]) -> Grid:
    return [[permutation[cell] for cell in row] for row in grid]


def _apply_color_permutation(task: dict[str, Any]) -> dict[str, Any]:
    permutation = list(range(10))
    random.shuffle(permutation)
    augmented = _copy_task(task)
    for split in ("train", "test"):
        for pair in augmented[split]:
            pair["input"] = _permute_grid_colors(pair["input"], permutation)
            pair["output"] = _permute_grid_colors(pair["output"], permutation)
    return augmented


def _all_nonzero_cells(task: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    cells: list[tuple[int, int, int, int]] = []
    for split in ("train", "test"):
        for pair in task[split]:
            for key in ("input", "output"):
                grid = pair[key]
                height = len(grid)
                width = len(grid[0])
                for row_idx, row in enumerate(grid):
                    for col_idx, color in enumerate(row):
                        if color != 0:
                            cells.append((row_idx, col_idx, height, width))
    return cells


def _translate_grid(grid: Grid, row_shift: int, col_shift: int) -> Grid:
    translated = [[0 for _ in row] for row in grid]
    for row_idx, row in enumerate(grid):
        for col_idx, color in enumerate(row):
            if color == 0:
                continue
            translated[row_idx + row_shift][col_idx + col_shift] = color
    return translated


def _apply_translation(task: dict[str, Any]) -> dict[str, Any]:
    cells = _all_nonzero_cells(task)
    if not cells:
        return _copy_task(task)
    min_row_shift = max(-row for row, _col, _height, _width in cells)
    max_row_shift = min(height - 1 - row for row, _col, height, _width in cells)
    min_col_shift = max(-col for _row, col, _height, _width in cells)
    max_col_shift = min(width - 1 - col for _row, col, _height, width in cells)
    shifts = [
        (row_shift, col_shift)
        for row_shift in range(min_row_shift, max_row_shift + 1)
        for col_shift in range(min_col_shift, max_col_shift + 1)
        if row_shift != 0 or col_shift != 0
    ]
    if not shifts:
        return _copy_task(task)
    row_shift, col_shift = random.choice(shifts)
    augmented = _copy_task(task)
    for split in ("train", "test"):
        for pair in augmented[split]:
            pair["input"] = _translate_grid(pair["input"], row_shift, col_shift)
            pair["output"] = _translate_grid(pair["output"], row_shift, col_shift)
    return augmented


def _add_grid_noise(grid: Grid, noise_prob: float = 0.02) -> Grid:
    noisy = _copy_grid(grid)
    for row_idx, row in enumerate(noisy):
        for col_idx, color in enumerate(row):
            if color == 0 and random.random() < noise_prob:
                noisy[row_idx][col_idx] = random.randint(1, 9)
    return noisy


def _apply_input_noise(task: dict[str, Any]) -> dict[str, Any]:
    augmented = _copy_task(task)
    for split in ("train", "test"):
        for pair in augmented[split]:
            pair["input"] = _add_grid_noise(pair["input"])
    return augmented


def augment_structured_task(
    task: dict[str, Any],
    color_permutation: bool = False,
    translation: bool = False,
    grid_noise: bool = False,
) -> list[dict[str, Any]]:
    """Return deterministic originals plus one-task augmentation variants."""
    variants = [_copy_task(task)]
    if color_permutation:
        variants.append(_apply_color_permutation(task))
    if translation:
        variants.append(_apply_translation(task))
    if grid_noise:
        variants.append(_apply_input_noise(task))
    return variants


def structured_example_difficulty(example: ArcStructuredExample) -> int:
    target_area = example.target_height * example.target_width
    context_area = len(example.context_colors)
    num_colors = len(set(example.context_colors) | set(example.target_colors))
    return target_area + context_area + (num_colors * 10)


def build_structured_examples_from_task(
    task_id: str,
    task: dict[str, Any],
    max_grid_size: int = 30,
    max_examples: int = 8,
) -> list[ArcStructuredExample]:
    """Build grid-native ARC examples from one task."""
    train_pairs: list[Pair] = [
        pair for pair in task["train"] if "input" in pair and "output" in pair
    ]
    test_pairs: list[Pair] = [pair for pair in task["test"] if "input" in pair and "output" in pair]
    raw_examples: list[tuple[list[Pair], Grid, Grid]] = []
    for idx, pair in enumerate(train_pairs):
        demos = [demo for demo_idx, demo in enumerate(train_pairs) if demo_idx != idx]
        raw_examples.append((demos, pair["input"], pair["output"]))
    for pair in test_pairs:
        raw_examples.append((train_pairs, pair["input"], pair["output"]))

    structured: list[ArcStructuredExample] = []
    for demos, query_input, output in raw_examples:
        if len(demos) > max_examples:
            continue
        try:
            serialize_grid(query_input)
            serialize_grid(output)
            for demo in demos:
                serialize_grid(demo["input"])
                serialize_grid(demo["output"])
        except ValueError:
            continue
        grids = [query_input, output]
        for demo in demos:
            grids.extend([demo["input"], demo["output"]])
        if any(not _grid_fits(grid, max_grid_size) for grid in grids):
            continue

        colors: list[int] = []
        rows: list[int] = []
        cols: list[int] = []
        roles: list[int] = []
        examples: list[int] = []
        object_ids: list[int] = []
        object_size_buckets: list[int] = []
        object_heights: list[int] = []
        object_widths: list[int] = []
        object_rel_rows: list[int] = []
        object_rel_cols: list[int] = []
        for demo_idx, demo in enumerate(demos):
            _append_grid_cells(
                demo["input"],
                ROLE_DEMO_INPUT,
                demo_idx,
                colors,
                rows,
                cols,
                roles,
                examples,
                object_ids,
                object_size_buckets,
                object_heights,
                object_widths,
                object_rel_rows,
                object_rel_cols,
            )
            _append_grid_cells(
                demo["output"],
                ROLE_DEMO_OUTPUT,
                demo_idx,
                colors,
                rows,
                cols,
                roles,
                examples,
                object_ids,
                object_size_buckets,
                object_heights,
                object_widths,
                object_rel_rows,
                object_rel_cols,
            )
        _append_grid_cells(
            query_input,
            ROLE_QUERY_INPUT,
            len(demos),
            colors,
            rows,
            cols,
            roles,
            examples,
            object_ids,
            object_size_buckets,
            object_heights,
            object_widths,
            object_rel_rows,
            object_rel_cols,
        )

        target_colors: list[int] = []
        target_rows: list[int] = []
        target_cols: list[int] = []
        for row_idx, row in enumerate(output):
            for col_idx, color in enumerate(row):
                target_colors.append(color)
                target_rows.append(row_idx)
                target_cols.append(col_idx)

        structured.append(
            ArcStructuredExample(
                task_id=task_id,
                context_colors=tuple(colors),
                context_rows=tuple(rows),
                context_cols=tuple(cols),
                context_roles=tuple(roles),
                context_examples=tuple(examples),
                context_object_ids=tuple(object_ids),
                context_object_size_buckets=tuple(object_size_buckets),
                context_object_heights=tuple(object_heights),
                context_object_widths=tuple(object_widths),
                context_object_rel_rows=tuple(object_rel_rows),
                context_object_rel_cols=tuple(object_rel_cols),
                target_colors=tuple(target_colors),
                target_rows=tuple(target_rows),
                target_cols=tuple(target_cols),
                target_height=len(output),
                target_width=len(output[0]),
            )
        )
    return structured


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


class ArcStructuredDataset(Dataset):
    """Map-style dataset of grid-native ARC examples."""

    def __init__(
        self,
        files: list[str | Path],
        max_grid_size: int = 30,
        max_examples: int = 8,
        augment_color_permutation: bool = False,
        augment_translation: bool = False,
        augment_grid_noise: bool = False,
        curriculum: bool = False,
    ):
        self.examples: list[ArcStructuredExample] = []
        self.skipped_tasks = 0
        for path in files:
            try:
                task = load_arc_task(path)
                examples: list[ArcStructuredExample] = []
                variants = augment_structured_task(
                    task,
                    color_permutation=augment_color_permutation,
                    translation=augment_translation,
                    grid_noise=augment_grid_noise,
                )
                for variant_idx, variant in enumerate(variants):
                    suffix = "" if variant_idx == 0 else f":aug{variant_idx}"
                    examples.extend(
                        build_structured_examples_from_task(
                            f"{Path(path).stem}{suffix}",
                            variant,
                            max_grid_size=max_grid_size,
                            max_examples=max_examples,
                        )
                    )
            except ValueError:
                self.skipped_tasks += 1
                continue
            self.examples.extend(examples)
        if curriculum:
            self.examples.sort(key=structured_example_difficulty)
        if not self.examples:
            raise ValueError("no structured ARC examples found")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ArcStructuredExample:
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


def collate_structured_arc_examples(
    batch: list[ArcStructuredExample],
) -> dict[str, torch.Tensor | list[str] | list[tuple[int, int]]]:
    """Pad grid-native ARC examples into tensors."""
    max_context = max(len(example.context_colors) for example in batch)
    max_target = max(len(example.target_colors) for example in batch)
    batch_size = len(batch)

    context_colors = torch.zeros((batch_size, max_context), dtype=torch.long)
    context_rows = torch.zeros((batch_size, max_context), dtype=torch.long)
    context_cols = torch.zeros((batch_size, max_context), dtype=torch.long)
    context_roles = torch.zeros((batch_size, max_context), dtype=torch.long)
    context_examples = torch.zeros((batch_size, max_context), dtype=torch.long)
    context_object_ids = torch.zeros((batch_size, max_context), dtype=torch.long)
    context_object_size_buckets = torch.zeros((batch_size, max_context), dtype=torch.long)
    context_object_heights = torch.zeros((batch_size, max_context), dtype=torch.long)
    context_object_widths = torch.zeros((batch_size, max_context), dtype=torch.long)
    context_object_rel_rows = torch.zeros((batch_size, max_context), dtype=torch.long)
    context_object_rel_cols = torch.zeros((batch_size, max_context), dtype=torch.long)
    context_mask = torch.zeros((batch_size, max_context), dtype=torch.bool)
    target_colors = torch.zeros((batch_size, max_target), dtype=torch.long)
    target_rows = torch.zeros((batch_size, max_target), dtype=torch.long)
    target_cols = torch.zeros((batch_size, max_target), dtype=torch.long)
    target_mask = torch.zeros((batch_size, max_target), dtype=torch.bool)
    target_heights = torch.zeros(batch_size, dtype=torch.long)
    target_widths = torch.zeros(batch_size, dtype=torch.long)
    task_ids: list[str] = []
    target_shapes: list[tuple[int, int]] = []

    for row, example in enumerate(batch):
        context_len = len(example.context_colors)
        target_len = len(example.target_colors)
        context_colors[row, :context_len] = torch.tensor(example.context_colors, dtype=torch.long)
        context_rows[row, :context_len] = torch.tensor(example.context_rows, dtype=torch.long)
        context_cols[row, :context_len] = torch.tensor(example.context_cols, dtype=torch.long)
        context_roles[row, :context_len] = torch.tensor(example.context_roles, dtype=torch.long)
        context_examples[row, :context_len] = torch.tensor(
            example.context_examples,
            dtype=torch.long,
        )
        context_object_ids[row, :context_len] = torch.tensor(
            example.context_object_ids,
            dtype=torch.long,
        )
        context_object_size_buckets[row, :context_len] = torch.tensor(
            example.context_object_size_buckets,
            dtype=torch.long,
        )
        context_object_heights[row, :context_len] = torch.tensor(
            example.context_object_heights,
            dtype=torch.long,
        )
        context_object_widths[row, :context_len] = torch.tensor(
            example.context_object_widths,
            dtype=torch.long,
        )
        context_object_rel_rows[row, :context_len] = torch.tensor(
            example.context_object_rel_rows,
            dtype=torch.long,
        )
        context_object_rel_cols[row, :context_len] = torch.tensor(
            example.context_object_rel_cols,
            dtype=torch.long,
        )
        context_mask[row, :context_len] = True
        target_colors[row, :target_len] = torch.tensor(example.target_colors, dtype=torch.long)
        target_rows[row, :target_len] = torch.tensor(example.target_rows, dtype=torch.long)
        target_cols[row, :target_len] = torch.tensor(example.target_cols, dtype=torch.long)
        target_mask[row, :target_len] = True
        target_heights[row] = example.target_height
        target_widths[row] = example.target_width
        task_ids.append(example.task_id)
        target_shapes.append((example.target_height, example.target_width))

    return {
        "context_colors": context_colors,
        "context_rows": context_rows,
        "context_cols": context_cols,
        "context_roles": context_roles,
        "context_examples": context_examples,
        "context_object_ids": context_object_ids,
        "context_object_size_buckets": context_object_size_buckets,
        "context_object_heights": context_object_heights,
        "context_object_widths": context_object_widths,
        "context_object_rel_rows": context_object_rel_rows,
        "context_object_rel_cols": context_object_rel_cols,
        "context_mask": context_mask,
        "target_colors": target_colors,
        "target_rows": target_rows,
        "target_cols": target_cols,
        "target_mask": target_mask,
        "target_heights": target_heights,
        "target_widths": target_widths,
        "task_ids": task_ids,
        "target_shapes": target_shapes,
    }
