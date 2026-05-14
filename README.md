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

Run the structured encoder architecture on an NVIDIA CUDA host:

```bash
uv run python src/rdlm/train_arc.py \
  --arch structured_encoder \
  --device cuda \
  --data-dir /path/to/arc/tasks \
  --eval-dir /path/to/arc/eval_tasks \
  --dim 256 \
  --batch-size 8 \
  --num-workers 4 \
  --steps 20000 \
  --checkpoint-dir checkpoints/arc
```

Run the serialized baseline for comparison:

```bash
uv run python src/rdlm/train_arc.py \
  --arch serialized \
  --device cuda \
  --data-dir /path/to/arc/tasks \
  --seq-len 1024 \
  --dim 256 \
  --batch-size 8 \
  --num-workers 4 \
  --steps 20000 \
  --checkpoint-dir checkpoints/arc_serialized
```

Use `--eval-dir` for held-out ARC JSON files with `test[*].output` populated.
Use `--resume checkpoints/arc/latest.pt` to continue a stopped run.
If `--device cuda` is requested on a host without CUDA-enabled PyTorch, the
trainer exits with an explicit error instead of falling back to CPU/MPS.

Structured encoder training also supports optional memory/regularization controls:
`--gradient-checkpointing`, `--aux-loss-weight`, `--stochastic-depth-prob`,
`--augment-color-permutation`, `--augment-translation`, `--augment-grid-noise`,
`--curriculum`, and `--use-object-features`.

Use `--eval-report report.json` to write structured evaluation metrics and
per-example rows as JSON. Use `--debug-dir debug/arc --debug-limit 5` to write
per-example predicted grids, confidence grids, and greedy diffusion trajectories.

Useful first ablations:

```bash
# Structured baseline
uv run python src/rdlm/train_arc.py --arch structured_encoder --eval-only \
  --eval-dir /path/to/arc/eval_tasks --resume checkpoints/arc/latest.pt \
  --eval-report reports/baseline.json

# Ensemble inference
uv run python src/rdlm/train_arc.py --arch structured_encoder --eval-only \
  --eval-dir /path/to/arc/eval_tasks --resume checkpoints/arc/latest.pt \
  --inference-mode ensemble --num-candidates 8 \
  --eval-report reports/ensemble.json

# Object-feature model path
uv run python src/rdlm/train_arc.py --arch structured_encoder --device cuda \
  --data-dir /path/to/arc/tasks --eval-dir /path/to/arc/eval_tasks \
  --use-object-features --checkpoint-dir checkpoints/arc_objects
```
