# rdlm — Recursive Diffusion Language Model

## ARC-AGI training

This repo can train the existing recursive diffusion LM on generic ARC-style JSON tasks.
ARC tasks are serialized as text-like token sequences: demonstrations and the query input
stay visible, and the model learns to denoise only the answer grid suffix.

Expected data shape is the standard ARC JSON structure:

```json
{
  "train": [{"input": [[1]], "output": [[2]]}],
  "test": [{"input": [[3]], "output": [[4]]}]
}
```

Run a tiny smoke check with a local fixture:

```bash
uv run python -m unittest discover -s tests
```

Run a serious local MPS/CPU training job against local ARC JSON files:

```bash
uv run python src/rdlm/train_arc.py \
  --data-dir /path/to/arc/tasks \
  --seq-len 1024 \
  --dim 256 \
  --batch-size 8 \
  --steps 20000 \
  --checkpoint-dir checkpoints/arc
```

Use `--eval-dir` for held-out ARC JSON files with `test[*].output` populated.
Use `--resume checkpoints/arc/latest.pt` to continue a stopped run.
