# Artifacts

This directory stores experiment results that are tracked in the git repository
so every commit is a reproducible record of what was trained and evaluated.

## Structure

```
artifacts/
  training/           ← Training summaries (auto-saved by train_arc.py)
    latest.json         Latest training summary with full args, model info, and metrics
    step_0000000.json   Step-specific summaries
  reports/            ← Evaluation reports (written by --eval-report)
    arc_eval.json        Full structured evaluation
    arc_eval_smoke.json  Quick smoke-test evaluation
    arc_debug/           Per-example debug payloads with grids and trajectories
    arc_debug_smoke/     Smoke-test debug payloads
```

## Commit Discipline

Every experiment that changes model behavior or evaluation results should be
committed so the git history fully documents the evolution.

- **Training summaries** (`artifacts/training/*.json`) capture the complete
  training configuration (all CLI flags), model parameter count, final loss,
  and the most recent evaluation metrics.
- **Evaluation reports** (`artifacts/reports/*.json`) capture per-example
  metrics, summary statistics, and full inference metadata (inference mode,
  sample steps, ensemble strategy, checkpoint path, etc.).

Together they guarantee that a checkout of any commit contains enough
information to understand exactly what was trained and how well it performed,
without relying on external experiment trackers or checkpoints.
