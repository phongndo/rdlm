# Actionable Research Synthesis for RDLM

Compiled: 2026-05-14

## Scope and corpus discovery

The active terminal was opened in `tau`, a terminal-emulator repo, but the AI project matching the request is the sibling repository `rdlm` (`/Users/joey/Programming/Node/rdlm`), a Recursive Diffusion Language Model for ARC-AGI.

Existing corpus and research infrastructure found:

- `research_findings.md` — curated review of ~50 masked diffusion LM, TRM/HRM, ARC, guidance, and test-time compute papers.
- `artifacts/research/raw_papers.jsonl` — 116 gathered metadata records from Semantic Scholar/OpenAlex/arXiv-style sources.
- `artifacts/research/ranked_papers.md` and `.scored.jsonl` — ranked bibliography and topic labels.
- `scripts/gather_research.py` — focused metadata gathering queries covering recursive reasoning, small LMs, diffusion text, test-time compute, latent CoT/planning, ARC, neuro-symbolic, state-space/sparse, curriculum/RL, and interpretability.
- `scripts/rank_research.py` — transparent keyword-based ranking against this codebase.
- `artifacts/reports/arc_eval.json`, `arc_eval_smoke.json`, `reports/arc_debug*` — existing ARC eval/debug reports.
- Core implementation: `src/rdlm/trm.py`, `tiny_block.py`, `diffusion_lm.py`, `arc.py`, `arc_model.py`, `train_arc.py`.

No broad external paper search was performed. This synthesis uses the existing local corpus as the primary source.

## Current repository baseline

RDLM is already a strong fit for the research corpus:

- Backbone: tiny recursive model (`TinyRecursiveModel`) repeatedly applies a shared `TinyBlock` with learnable output/latent states, timestep conditioning, and halting head.
- ARC path: grid-native encoder-conditioned output diffusion (`ArcOutputDiffusion`) with color/row/col/role/example embeddings and optional object features + shape head.
- Training: masked diffusion over only answer/output cells; structured ARC dataset supports color permutation, translation, input noise, and curriculum flags.
- Inference: greedy or ensemble diffusion over output grids, optional inferred-shape candidate generation, temporal voting, DOS-like sampling, calibration, adaptive sample steps, early stop, ARC heuristic reranking.
- Evaluation: ARC exact match, cell accuracy, nonzero IoU, color histogram L1, confidence/log-prob, valid output, shape exact/top-k/oracle shape.

Existing eval reports show `exact=0.0` on small eval subsets, with nontrivial cell accuracy (`~45.2%` on 10 examples under oracle target shape). Shape is currently oracle-perfect when target shape is supplied; inferred-shape remains the harder relevant setting.

Recommended primary metric for experiments: `exact` on held-out ARC eval with `--infer-shape` enabled once shape candidates are mature. During smoke/local iteration use `cell_acc` and `nonzero_iou` as fast secondary metrics, but do not optimize exclusively for them.

Quick correctness command verified locally:

```bash
uv run python -m unittest discover -s tests
```

## Literature map and topic clustering

### Recursive reasoning / HRM / TRM / recurrent inference

Most relevant: TRM, HRM, denoising recursion models, symbol-equivariant recurrent reasoning, dynamical systems views of HRM, Tiny AR recursive models, Mamba-2/TRM hybrids.

Core pattern: small shared modules gain effective depth through recurrent/recursive application. The most applicable insight is not just “loop more”; it is to train the loop trajectory so intermediate states become increasingly useful, stable, and inspectable.

### Masked/discrete diffusion language modeling

Most relevant: MDLM, dLLM, Scaling Beyond MDLM/SID, DiffusionBERT, simplified/generalized masked diffusion, discrete guidance.

Core pattern: random independent masking is simple and stable, but can mismatch inference trajectories. Loss weighting, schedule design, and structured noising are high-leverage.

### Decoding order, test-time compute, self-consistency

Most relevant: Where-to-Unmask, DOS, DUS, TRIMS, MDPO, No Compute Left Behind, Time Is a Feature, UnMaskFork, SMC for MDLMs.

Core pattern: confidence-only reveal order is often brittle. Better methods use dependency, role, temporal stability, search, or supervision over reveal trajectories.

### ARC / neuro-symbolic / object-centric reasoning

Most relevant: ARC-AGI surveys, CompressARC, ARC-TGI, executable world models, object-centric/grid-prior approaches.

Core pattern: ARC generalization needs strong invariances, object/shape priors, hypothesis/candidate verification, and data generation. Pure token likelihood is not sufficient.

### Efficient sequence models / memory / sparse architectures

Most relevant: Mamba/selective SSMs, Mamba-TRM hybrids, sparse/state-space efficient transformer literature.

Core pattern: recurrent state models may fit recursive refinement better than full attention for small models, but dependencies and implementation risk are high.

### Curriculum / synthetic data / RL / verifier systems

Most relevant: ARC-TGI, d1/diffusion RL, consistency-trajectory RL, self-rewarding SMC, verifier/process-supervision papers.

Core pattern: low-data reasoning gains come from curriculum + generated tasks + process/reward signal, not only architecture.

## Ranked most useful papers/findings for this codebase

1. **Less is More: Recursive Reasoning with Tiny Networks (TRM)** — foundational architecture. Current code implements the key recursive latent/output refinement idea. Next relevance: deeper trajectory supervision and adaptive loop depth.
2. **Simple and Effective Masked Diffusion Language Models (MDLM)** — validates current masked diffusion direction. Next relevance: objective/loss weighting and schedule ablations.
3. **MDPO / training-inference alignment for MDLMs** — directly targets current random-masking vs structured-unmasking mismatch.
4. **Time Is a Feature** — already partially reflected via temporal voting; should be evaluated more systematically and used as a diagnostic.
5. **Where-to-Unmask / TRIMS** — supports adding an auxiliary reveal-order head and supervised trajectory targets.
6. **DOS / dependency-oriented sampling** — low-risk decoding change; code already exposes a `--sampling-strategy dos` path that should be benchmarked under inferred-shape eval.
7. **No Compute Left Behind / Parallelism and Generation Order in MDLMs** — cautions against excessive parallel reveal; supports left-to-right/structured reveal ablations for ARC grids.
8. **CompressARC / MDL principle** — strong ARC-specific candidate-selection prior, highly relevant to reranking sampled grids.
9. **ARC-TGI / task generators** — likely highest sample-efficiency lever after inference fixes.
10. **Symbol-Equivariant Recurrent Reasoning Models** — directly relevant to ARC color permutation and symbol equivariance; suggests architectural equivariance beyond augmentation.
11. **Mamba / Mamba-TRM hybrid** — promising efficiency/recurrence direction, but higher implementation risk.
12. **Simple guidance for discrete diffusion** — could guide toward palette, object, density, or verifier constraints.

## Key recurring architectural patterns

- **Shared-weight iterative refinement**: improve effective depth without increasing parameters.
- **Dual state streams**: separate latent/reasoning state from output state; current TRM already uses this.
- **Train/inference trajectory matching**: training masks should resemble inference states, not independent Bernoulli masks only.
- **Order matters**: reveal/correction order should encode dependencies, roles, spatial locality, or temporal stability.
- **Temporal diagnostics**: intermediate denoising states contain useful candidates; late steps can overwrite correct cells.
- **Candidate generation + verifier/reranker**: small models benefit from cheap test-time search plus task-specific verification.
- **Equivariance/invariance**: color symbols, translations, rotations/reflections, and object identity are central to ARC generalization.
- **Object/shape factoring**: separate shape/object/layout inference from color fill-in.
- **Curriculum and synthetic tasks**: sample efficiency depends on exposing compositional transformations, not only real ARC tasks.
- **Adaptive compute**: allocate more steps/branches to uncertain, high-entropy, large, or inconsistent grids.

## Gaps and contradictions in the corpus

- **Diffusion parallelism vs reasoning quality**: some MDLM papers emphasize parallel decoding speed, while reasoning papers report left-to-right or dependency-ordered decoding can outperform any-order decoding.
- **Confidence is both useful and misleading**: confidence identifies easy cells but can defer high-entropy logical cells and reinforce wrong early guesses.
- **Scaling claims are not ARC-specific**: large language-model findings often do not transfer directly to small ARC models.
- **Synthetic data helps but can overfit generator biases**: ARC-TGI-style generators must be held out by transformation family, not mixed freely into eval.
- **Object features are plausible but not guaranteed**: object-centric priors can fail on tasks where background or color roles are relational rather than connected-component based.
- **State-space/Mamba swaps are promising but risky**: may improve efficiency, but dependency and implementation complexity conflict with minimal ablation-friendly changes.

## Prioritized implementation experiments

### 1. Standardize inferred-shape ARC eval harness

- **Hypothesis**: Current oracle-shape eval overestimates reasoning progress; inferred-shape exact match is the right target.
- **Expected mechanism**: Separates shape inference, cell denoising, and candidate ranking.
- **Implementation plan**: Add a small `scripts/eval_arc_smoke.sh` or document canonical command using `--infer-shape --shape-top-k 5 --dump-candidates --eval-limit N`.
- **Files likely affected**: `README.md`, possibly `scripts/`.
- **Evaluation strategy**: Track `exact`, `shape_exact`, `shape_topk_hit`, `cell_acc`, `nonzero_iou` on fixed eval subsets.
- **Expected benefit**: Prevents optimizing the wrong metric.
- **Computational cost**: Low.
- **Risks/failure modes**: Local machine may lack CUDA/checkpoints/datasets.
- **Difficulty**: Easy.

### 2. Benchmark existing DOS, temporal voting, calibration, and adaptive steps factorially

- **Hypothesis**: Some already-implemented inference flags provide gains, but interactions are unknown.
- **Expected mechanism**: Temporal stability and dependency-aware reveal reduce high-confidence wrong cells.
- **Implementation plan**: Run fixed-seed eval grid over `--sampling-strategy confidence|dos`, `--temporal-vote`, `--enable-calibration`, `--adaptive-sample-steps`, under oracle and inferred shape.
- **Files likely affected**: none, or `scripts/run_eval_grid.py`.
- **Evaluation strategy**: Compare exact/cell_acc/nonzero_iou/latency using same checkpoint and eval subset.
- **Expected benefit**: High; may reveal free wins.
- **Computational cost**: Low-medium depending eval size.
- **Risks/failure modes**: Small eval subsets noisy; exact may remain zero.
- **Difficulty**: Easy.

### 3. MDPO-style structured masking during training

- **Hypothesis**: Training with masks sampled from inference-like partial reveal states improves denoising trajectory quality.
- **Expected mechanism**: Reduces random-mask/train vs confidence-reveal/inference mismatch.
- **Implementation plan**: Add a `--mask-schedule {bernoulli,cosine_reveal,block,spatial}` option in `ArcOutputDiffusion._mask_targets`; sample mask sets matching cosine transfer counts and/or spatial reveal order.
- **Files likely affected**: `src/rdlm/arc_model.py`, `src/rdlm/train_arc.py`, tests.
- **Evaluation strategy**: Train short runs with same seed/steps; eval exact/cell_acc and trajectory stability.
- **Expected benefit**: High.
- **Computational cost**: Medium; retraining required.
- **Risks/failure modes**: Less noise diversity; overfits to one sampler.
- **Difficulty**: Medium.

### 4. Supervised reveal-order auxiliary head

- **Hypothesis**: A learned reveal priority can outperform confidence-only selection.
- **Expected mechanism**: Head learns which cells are structurally decisive, not merely easy.
- **Implementation plan**: Add per-cell priority logits to `ArcOutputDiffusion`; derive targets from ground-truth change/error under partially masked states or deterministic spatial dependencies; use small aux loss.
- **Files likely affected**: `arc_model.py`, `train_arc.py`.
- **Evaluation strategy**: Compare reveal-order ablation vs confidence/DOS on same checkpoint training budget.
- **Expected benefit**: High.
- **Computational cost**: Medium.
- **Risks/failure modes**: Bad target construction could teach trivial order.
- **Difficulty**: Medium-hard.

### 5. Temporal consistency as training regularizer

- **Hypothesis**: Penalizing prediction flips across adjacent noising levels improves stable reasoning.
- **Expected mechanism**: Encourages coherent denoising trajectories and reduces late overwrites.
- **Implementation plan**: During training, evaluate two nearby timesteps/masks and add KL or consistency loss on eligible target cells.
- **Files likely affected**: `arc_model.py`, `train_arc.py`.
- **Evaluation strategy**: Track flip rate in debug trajectories plus exact/cell_acc.
- **Expected benefit**: Medium-high.
- **Computational cost**: ~2x forward unless optimized.
- **Risks/failure modes**: Over-smoothing, reduced correction ability.
- **Difficulty**: Medium.

### 6. MDL/compression-inspired candidate reranker

- **Hypothesis**: ARC candidates with simpler transformations/program descriptions generalize better than highest likelihood candidates.
- **Expected mechanism**: Penalizes arbitrary color noise and rewards compressible grids consistent with demos.
- **Implementation plan**: Extend `arc_heuristic_candidate_score` with simple description length terms: number of colors, connected components, symmetry, repeated rows/cols, bounding boxes, affine relation to input/demo outputs.
- **Files likely affected**: `train_arc.py`, maybe new `src/rdlm/arc_heuristics.py`.
- **Evaluation strategy**: Use `--dump-candidates`; evaluate oracle-in-candidate and selected-candidate accuracy.
- **Expected benefit**: Medium-high, especially under ensemble/inferred shape.
- **Computational cost**: Low.
- **Risks/failure modes**: Hand-coded priors may hurt tasks requiring complex outputs.
- **Difficulty**: Easy-medium.

### 7. Color-symbol equivariant architecture ablation

- **Hypothesis**: Architectural color equivariance beats augmentation-only color permutation.
- **Expected mechanism**: Prevents memorizing absolute color identities where ARC requires relational roles.
- **Implementation plan**: Test color dropout/permutation during every batch; later consider factorizing color embeddings into learned symbol role + palette context.
- **Files likely affected**: `arc.py`, `arc_model.py`.
- **Evaluation strategy**: Evaluate on color-permuted validation copy and standard eval.
- **Expected benefit**: Medium-high for generalization.
- **Computational cost**: Low-medium.
- **Risks/failure modes**: Some ARC tasks use semantic color constants; full equivariance may remove useful cues.
- **Difficulty**: Medium.

### 8. Object-feature ablation matrix

- **Hypothesis**: Connected-component features help object-transform tasks but may hurt pixel-pattern tasks.
- **Expected mechanism**: Supplies object identity, size, bounding-box-relative positions.
- **Implementation plan**: Train/eval with `--use-object-features`, with and without augmentations and shape head.
- **Files likely affected**: none unless adding per-domain metrics.
- **Evaluation strategy**: Bucket eval tasks by component count/size; compare metrics by bucket.
- **Expected benefit**: Medium.
- **Computational cost**: Medium.
- **Risks/failure modes**: Feature leakage/bias; extra embeddings overfit.
- **Difficulty**: Easy if training infra available.

### 9. Learned shape head under true inferred-shape eval

- **Hypothesis**: A trained shape head plus context priors improves `shape_exact` and downstream exact match.
- **Expected mechanism**: Separates output canvas prediction from color denoising.
- **Implementation plan**: Train with `--use-shape-head --shape-loss-weight`; tune `--shape-score-weight`; inspect top-k coverage.
- **Files likely affected**: none initially.
- **Evaluation strategy**: `shape_exact`, `shape_topk_hit`, `oracle_shape_exact`, final `exact`.
- **Expected benefit**: High if shape is bottleneck.
- **Computational cost**: Medium.
- **Risks/failure modes**: Shape prior from demos may outperform neural head; bad shape scoring can select wrong canvas.
- **Difficulty**: Easy-medium.

### 10. Adaptive compute based on uncertainty and area

- **Hypothesis**: Fixed 64 denoising steps wastes compute on easy grids and under-computes hard grids.
- **Expected mechanism**: More steps for large/high-entropy/unstable candidates; early stop easy candidates.
- **Implementation plan**: Expand existing `--adaptive-sample-steps` and early-stop criteria to use running entropy/flip-rate, not area only.
- **Files likely affected**: `arc_model.py`, `train_arc.py`.
- **Evaluation strategy**: exact/cell_acc vs mean steps/latency.
- **Expected benefit**: Medium.
- **Computational cost**: Low implementation; can lower runtime.
- **Risks/failure modes**: Early stopping locks wrong candidates.
- **Difficulty**: Easy-medium.

### 11. Candidate verifier head

- **Hypothesis**: A lightweight verifier trained on generated/perturbed outputs improves reranking.
- **Expected mechanism**: Scores demo-consistency and output plausibility independently of denoising likelihood.
- **Implementation plan**: Generate positive true outputs and negative corrupted/sample outputs; train context+candidate binary head.
- **Files likely affected**: `arc_model.py`, `train_arc.py`, data generation utilities.
- **Evaluation strategy**: Candidate selection accuracy and final exact under fixed candidate pool.
- **Expected benefit**: High.
- **Computational cost**: Medium-high.
- **Risks/failure modes**: Verifier overfits superficial statistics; false confidence.
- **Difficulty**: Hard.

### 12. ARC-TGI-style synthetic curriculum

- **Hypothesis**: Transformation-family synthetic tasks improve sample efficiency and held-out generalization.
- **Expected mechanism**: Covers transformations underrepresented in ARC train set with controlled curricula.
- **Implementation plan**: Add small deterministic generators for copy, recolor, translate, crop, mirror, object count, fill bounding box; hold out generator families for validation.
- **Files likely affected**: new `src/rdlm/arc_generators.py`, `arc.py`, `train_arc.py`.
- **Evaluation strategy**: Synthetic held-out family accuracy + real ARC eval.
- **Expected benefit**: High.
- **Computational cost**: Medium.
- **Risks/failure modes**: Generator bias; synthetic gains do not transfer.
- **Difficulty**: Medium-hard.

### 13. Soft-masked target embeddings

- **Hypothesis**: Soft mixing predicted/token embeddings by confidence improves gradient flow and reduces hard mask discontinuities.
- **Expected mechanism**: Model sees uncertainty as continuous state instead of binary mask/reveal.
- **Implementation plan**: During denoising, feed expected color embedding or confidence-weighted mask/color blend for target cells; start as inference-only ablation, then train-compatible path.
- **Files likely affected**: `arc_model.py`, `diffusion_lm.py`, `trm.py` for text path if generalized.
- **Evaluation strategy**: Compare trajectory stability and final metrics.
- **Expected benefit**: Medium.
- **Computational cost**: Low-medium.
- **Risks/failure modes**: Train/inference mismatch unless trained with soft states.
- **Difficulty**: Medium.

### 14. Relative/2D positional relaxation

- **Hypothesis**: Absolute row/col embeddings reduce robustness to translations and shape shifts.
- **Expected mechanism**: Relative or normalized 2D positions improve spatial generalization.
- **Implementation plan**: Add optional relative row/col bias or normalized coordinate embeddings; compare with existing translation augmentation.
- **Files likely affected**: `arc_model.py`, `tiny_block.py` if attention bias added.
- **Evaluation strategy**: translation-perturbed validation and standard ARC.
- **Expected benefit**: Medium.
- **Computational cost**: Medium.
- **Risks/failure modes**: Attention-bias changes may destabilize small model.
- **Difficulty**: Medium-hard.

### 15. Left-to-right / scanline / object-order reveal ablation

- **Hypothesis**: For ARC, deterministic structured reveal can beat fully any-order confidence reveal.
- **Expected mechanism**: Provides stable dependencies similar to autoregressive decoding without changing model architecture.
- **Implementation plan**: Add `--sampling-strategy scanline|object|border-first|center-first` for target reveal priority.
- **Files likely affected**: `arc_model.py`, `train_arc.py`.
- **Evaluation strategy**: Compare to confidence/DOS under same steps.
- **Expected benefit**: Medium.
- **Computational cost**: Low.
- **Risks/failure modes**: One order will not suit all tasks.
- **Difficulty**: Easy-medium.

### 16. Mamba/SSM tiny block prototype

- **Hypothesis**: Selective state-space recurrence complements TRM refinement at lower cost than attention.
- **Expected mechanism**: Efficient recurrent memory across serialized/grid tokens.
- **Implementation plan**: Prototype behind `--block-type attention|ssm`; avoid new dependencies initially by implementing a minimal gated convolution/state update, not full Mamba.
- **Files likely affected**: `tiny_block.py`, `trm.py`, `arc_model.py`, `train_arc.py`.
- **Evaluation strategy**: Param count, train speed, eval metrics under small budget.
- **Expected benefit**: Speculative medium-high.
- **Computational cost**: Medium-high.
- **Risks/failure modes**: Reimplementation may underperform real Mamba; dependency creep.
- **Difficulty**: Hard.

## Prioritized roadmap

### Phase 0 — Measurement discipline

1. Canonicalize inferred-shape eval command and fixed small/medium eval subsets.
2. Run factorial benchmark over existing inference flags.
3. Add per-task bucket reporting: shape mismatch, object count, grid area, color count, trajectory flip rate.

### Phase 1 — Low-risk inference wins

1. Improve MDL/ARC heuristic reranker.
2. Expand adaptive compute from area-only to entropy/flip-rate.
3. Add deterministic reveal-order ablations.
4. Tune shape candidate scoring and learned shape head.

### Phase 2 — Training/inference alignment

1. Add structured masking schedules.
2. Add temporal consistency regularization.
3. Add reveal-order auxiliary head.
4. Evaluate color equivariance and object features with controlled ablations.

### Phase 3 — Data and verification

1. Build small ARC synthetic generator suite with held-out families.
2. Train a candidate verifier head.
3. Add process/candidate supervision from generated trajectories.

### Phase 4 — Higher-risk architecture

1. Soft-masked target embeddings.
2. Relative/2D positional relaxation.
3. Minimal SSM/Mamba-like block prototype only after baseline eval is stable.

## Immediate next command recommendation

If CUDA/checkpoints/datasets are available, start by measuring already-implemented inference options before changing architecture:

```bash
uv run python src/rdlm/train_arc.py \
  --arch structured_encoder --eval-only \
  --eval-dir /home/joey/Programming/arc-agi/data/evaluation \
  --resume checkpoints/arc_encoder/latest.pt \
  --infer-shape --shape-top-k 5 --dump-candidates \
  --eval-limit 20 --sample-steps 64 \
  --inference-mode ensemble --num-candidates 8 \
  --sampling-strategy dos --temporal-vote --enable-calibration \
  --eval-report artifacts/reports/infer_shape_dos_temporal_calibrated.json \
  --debug-dir artifacts/reports/debug_infer_shape_dos_temporal_calibrated
```

For training:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python src/rdlm/train_arc.py \
  --arch structured_encoder \
  --device cuda \
  --data-dir /home/joey/Programming/arc-agi/data/training \
  --eval-dir /home/joey/Programming/arc-agi/data/evaluation \
  --use-shape-head \
  --shape-loss-weight 0.1 \
  --dim 512 \
  --max-examples 2 \
  --gradient-checkpointing \
  --batch-size 1 \
  --lr 3e-4 \
  --steps 20000 \
  --eval-every 1000 \
  --eval-limit 50 \
  --sample-steps 32 \
  --save-every 1000 \
  --checkpoint-dir checkpoints/arc_infer_shape \
  --keep-latest-backups 10 \
  --best-checkpoint-metric exact \
  --eval-report artifacts/reports/arc_infer_shape_train_eval.json \
  --debug-dir artifacts/reports/arc_infer_shape_debug
```

If running locally without CUDA/checkpoints, keep using:

```bash
uv run python -m unittest discover -s tests
```
