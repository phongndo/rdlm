# Diffusion Learning Model Research Findings

Comprehensive literature review for **RDLM (Recursive Diffusion Language Model)** — a masked diffusion LM built on a TinyRecursiveModel (TRM) backbone, applied to ARC-AGI tasks.

> Compiled: 2026-05-14 | ~50 papers surveyed across arxiv (2022–2026)

---

## Table of Contents

1. [Core MDLM Papers](#1-core-mdlm-papers)
2. [Architectural Improvements for MDLMs](#2-architectural-improvements-for-mdlms)
3. [Unmasking Order / Decoding Strategies](#3-unmasking-order--decoding-strategies)
4. [Inference Acceleration for dLLMs](#4-inference-acceleration-for-dllms)
5. [Training–Inference Alignment & RL for MDLMs](#5-training-inference-alignment--rl-for-mdlms)
6. [Scaling Laws & Comparisons](#6-scaling-laws--comparisons)
7. [Recursive/Iterative Architectures (TRM Family)](#7-recursiveiterative-architectures-trm-family)
8. [ARC-AGI Benchmarks & Approaches](#8-arc-agi-benchmarks--approaches)
9. [Guidance & Control for Discrete Diffusion](#9-guidance--control-for-discrete-diffusion)
10. [Discrete Diffusion Foundations](#10-discrete-diffusion-foundations)
11. [Quick Wins for RDLM](#11-quick-wins-for-rdlm)

---

## 1. Core MDLM Papers

### Simple and Effective Masked Diffusion Language Models (MDLM)
- **Link:** [arxiv 2406.07524](https://arxiv.org/abs/2406.07524)
- **Authors:** Sahoo, Arriola, Schiff, Gokaslan, Marroquin, Chiu, Rush, Kuleshov (2024)
- **Key contribution:** Showed that simple masked discrete diffusion is more performant than previously thought. Introduced an effective training recipe and a simplified, Rao-Blackwellized objective that improves performance.
- **Relevance to RDLM:** Your `diffusion_lm.py` already references MDLM. This is the foundational paper for your approach.

### dLLM: Simple Diffusion Language Modeling
- **Link:** [arxiv 2602.22661](https://arxiv.org/abs/2602.22661)
- **Authors:** Zhou, Chen, Tong, Song (2026)
- **Key contribution:** Unified framework standardizing shared components in diffusion LMs. Provides transparent implementations of common architectures.
- **Relevance:** Directly referenced in your `diffusion_lm.py`. Worth studying their implementation patterns.

### Scaling Beyond Masked Diffusion Language Models
- **Link:** [arxiv 2602.15014](https://arxiv.org/abs/2602.15014)
- **Authors:** Sahoo, Lemercier, Yang, Deschenaux, Liu, Thickstun, Jukic (2026)
- **Key contribution:** First scaling law study for uniform-state and interpolating discrete diffusion methods. Shows MDLMs can be ~12% more FLOPs-efficient with an appropriate loss weighting scheme. Proposes "simple interpolating diffusion" (SID) that unifies uniform-state and masked approaches.
- **Relevance:** Important for understanding how to scale your model and which diffusion formulation to use.

### DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models
- **Link:** [arxiv 2211.15029](https://arxiv.org/abs/2211.15029)
- **Authors:** He, Sun, Wang, Huang, Qiu (2022)
- **Key contribution:** Combines BERT-style masked LMs with diffusion processes. Introduces a new noise schedule and training objective for masked diffusion.
- **Relevance:** Early foundational work bridging BERT pretraining with diffusion.

---

## 2. Architectural Improvements for MDLMs

### Soft-Masked Diffusion Language Models
- **Link:** [arxiv 2510.17206](https://arxiv.org/abs/2510.17206)
- **Authors:** Hersche, Moor-Smith, Hofmann, Rahimi (2025)
- **Key contribution:** Replaces binary mask-or-predict with a soft mixing of masked and predicted token embeddings weighted by confidence. Outperforms standard MDLMs with fewer decoding steps.
- **Relevance:** ⭐ **High impact for RDLM.** Soft masking could replace your hard mask-or-reveal mechanism in the backbone, potentially improving gradient flow and sample quality.

### Relaxing Positional Alignment in Masked Diffusion Language Models
- **Link:** [arxiv 2601.22947](https://arxiv.org/abs/2601.22947)
- **Authors:** Ye, Takahashi, Kudo, Suzuki (2026)
- **Key contribution:** Shows that strict positional prediction makes MDLM decoding highly sensitive to token misalignment (one-position shift can severely disrupt semantics). Proposes methods to relax positional alignment.
- **Relevance:** ⭐ **High impact for RDLM.** Your model uses absolute position embeddings — this paper's findings could help you improve robustness. Consider relative position encodings or their relaxation technique.

### Beyond Masks: Deletion-Insertion Diffusion Language Models (DID)
- **Link:** [arxiv 2603.23507](https://arxiv.org/abs/2603.23507)
- **Authors:** Ding, Ding, Chen, Wang, Xu, Feng, Bai, Han, Yan, Yuan, Sun (2026)
- **Key contribution:** Replaces masking/unmasking with deletion and insertion as discrete diffusion processes. Improves training and inference efficiency compared to MDLMs.
- **Relevance:** Alternative paradigm to masking — could lead to more flexible generation.

---

## 3. Unmasking Order / Decoding Strategies

### Where-to-Unmask: Ground-Truth-Guided Unmasking Order Learning
- **Link:** [arxiv 2602.09501](https://arxiv.org/abs/2602.09501)
- **Authors:** Asano, Kozuno, Saito, Baba (2026)
- **Key contribution:** Learns unmasking order directly from ground-truth data using a lightweight prediction head, avoiding expensive RL rollouts. Trained on a simple auxiliary loss.
- **Relevance:** ⭐ **High impact for RDLM.** Your model uses confidence-based unmasking (a heuristic). This paper shows how to learn the unmasking order explicitly, which could significantly improve generation quality.

### DOS: Dependency-Oriented Sampler for Masked Diffusion Language Models
- **Link:** [arxiv 2603.15340](https://arxiv.org/abs/2603.15340)
- **Authors:** Zhou, Hu, Huang (2026)
- **Key contribution:** Training-free decoding strategy that leverages sequence-level information and inter-token dependencies, unlike existing token-level uncertainty criteria.
- **Relevance:** Direct improvement over your confidence-based top-k sampling in `_sample_with_scores`. Easy to integrate.

### Plan for Speed: Dilated Scheduling for MDLMs
- **Link:** [arxiv 2506.19037](https://arxiv.org/abs/2506.19037)
- **Authors:** Luxembourg, Permuter, Nachmani (2025)
- **Key contribution:** Dilated Unmasking Scheduler (DUS) partitions positions into non-adjacent groups and unmasks them in parallel to minimize interaction misses.
- **Relevance:** Your model uses block-based generation — DUS could improve parallel unmasking within blocks.

### TRIMS: Trajectory-Ranked Instruction Masked Supervision
- **Link:** [arxiv 2604.00666](https://arxiv.org/abs/2604.00666)
- **Authors:** Chen, Qiu, Fan, Zhao, Tong (2026)
- **Key contribution:** Provides explicit supervision over token reveal order during training, fixing the train-inference mismatch in decoding trajectories.
- **Relevance:** Directly addresses a key weakness of your current training approach (random masking at training vs. structured unmasking at inference).

### No Compute Left Behind: Rethinking Reasoning and Sampling with MDLMs
- **Link:** [arxiv 2510.19990](https://arxiv.org/abs/2510.19990)
- **Authors:** Horvitz, Singhal, Zou, Domingo-Enrich, Yu, Ranganath, McKeown (2025)
- **Key contribution:** Shows that for math and coding tasks, any-order decoding often underperforms left-to-right sampling, and standard multi-token decoding degrades performance. Proposes using the full sequence of diffusion step predictions.
- **Relevance:** ⭐ **Very important read.** Their findings about when to use left-to-right vs. any-order directly apply to your ARC-AGI tasks.

### UnMaskFork: Test-Time Scaling for Masked Diffusion via Deterministic Action Branching
- **Link:** [arxiv 2602.04344](https://arxiv.org/abs/2602.04344)
- **Authors:** Misaki, Akiba (2026)
- **Key contribution:** Formulates the unmasking trajectory as a search tree using Monte Carlo Tree Search (MCTS) to improve generation quality at test time.
- **Relevance:** Your ensemble method in `ArcOutputDiffusion` could be extended to MCTS-based search for better candidates.

---

## 4. Inference Acceleration for dLLMs

### Not All Denoising Steps Are Equal: Model Scheduling for Faster MDLMs
- **Link:** [arxiv 2604.02340](https://arxiv.org/abs/2604.02340)
- **Authors:** Sedykh, Sorokin, Malykh (2026)
- **Key contribution:** Uses a smaller MDLM to replace the full model at a subset of denoising steps, exploiting the flexibility of diffusion.
- **Relevance:** ⭐ **High impact for RDLM.** Your TRM backbone is already small, but model scheduling could enable even faster ARC-AGI inference.

### ES-dLLM: Efficient Inference for dLLMs by Early-Skipping
- **Link:** [arxiv 2603.10088](https://arxiv.org/abs/2603.10088)
- **Authors:** Zhu, Ren, Tan, Ma (2026)
- **Key contribution:** Finds that intermediate representations (KV, hidden states) change only subtly between adjacent denoising steps. Proposes early-skipping to bypass computation for unchanged positions.
- **Relevance:** Could accelerate your inference — most denoising steps only change a few positions.

### R²-dLLM: Spatio-Temporal Redundancy Reduction
- **Link:** [arxiv 2604.18995](https://arxiv.org/abs/2604.18995)
- **Authors:** Du, Xia, Zhong, Fu, Oswald, Ji, Khailany, Molchanov, Lin (2026)
- **Key contribution:** Identifies spatial redundancy (confidence clusters, positional ambiguity) and temporal redundancy (repeated tokens across steps). Proposes early exiting and adaptive computation.
- **Relevance:** Similar to ES-dLLM — combined they could give ~2-4x speedup on your inference.

### Time Is a Feature: Exploiting Temporal Dynamics in dLLMs
- **Link:** [arxiv 2508.09138](https://arxiv.org/abs/2508.09138)
- **Authors:** Wang, Fang, Jing, Shen, Shen, Wang, Ouyang, Chen, Shen (2025)
- **Key contribution:** Discovers "temporal oscillation" — correct answers emerge mid-process but get overwritten. Proposes Temporal Self-Consistency Voting and Temporal Output Reweighting to exploit this.
- **Relevance:** ⭐ **High impact for RDLM.** Your confidence-based unmasking could be improved by tracking temporal consistency. The oscillation phenomenon is directly observable in your diffusion traces.

### Streaming-dLLM: Suffix Pruning and Dynamic Decoding
- **Link:** [arxiv 2601.17917](https://arxiv.org/abs/2601.17917)
- **Authors:** Xiao, Hao, Guo, Luo, Liu, Xu, Hu (2026)
- **Key contribution:** Accelerates dLLM decoding via suffix pruning and dynamic decoding strategies.
- **Relevance:** Useful for your text diffusion model in `train_diffusion.py`.

---

## 5. Training–Inference Alignment & RL for MDLMs

### MDPO: Overcoming the Training-Inference Divide of MDLMs
- **Link:** [arxiv 2508.13148](https://arxiv.org/abs/2508.13148)
- **Authors:** He, Renz, Cao, Geiger (2025)
- **Key contribution:** During inference, MDLMs progressively reveal structure by producing fewer masked tokens — but training ignores this structure (tokens masked at random). Proposes to bridge this gap.
- **Relevance:** ⭐ **Critical for RDLM.** Your `_apply_mask` method uses independent random masking per position, which is exactly the disconnect this paper addresses.

### d1: Scaling Reasoning in Diffusion LLMs via Reinforcement Learning
- **Link:** [arxiv 2504.12216](https://arxiv.org/abs/2504.12216)
- **Authors:** Zhao, Gupta, Zheng, Grover (2025)
- **Key contribution:** First paper to apply online RL (similar to DeepSeek-R1) to diffusion LLMs. Demonstrates improved reasoning by using RL to optimize the coarse-to-fine generation process.
- **Relevance:** ⭐ **High impact for RDLM.** Your ARC-AGI model could benefit from RL fine-tuning of the denoising trajectory.

### Taming MDLMs via Consistency Trajectory Reinforcement Learning
- **Link:** [arxiv 2509.23924](https://arxiv.org/abs/2509.23924)
- **Authors:** Yang, Chen, Hu, Shao (2025)
- **Key contribution:** RL approach for MDLMs that optimizes the decoding policy to achieve better quality with fewer steps.
- **Relevance:** Complementary to d1 — focuses on step-efficiency.

### Self-Rewarding Sequential Monte Carlo for MDLMs
- **Link:** [arxiv 2602.01849](https://arxiv.org/abs/2602.01849)
- **Authors:** Luo, Jin, Wang, Bing, Schön (2026)
- **Key contribution:** Inference-time scaling algorithm using SMC to overcome the greedy, noise-sensitive decoding of confidence-based sampling. Improves diversity and quality.
- **Relevance:** Could replace your confidence-based unmasking with a more principled particle filtering approach.

### DiffCoT: Diffusion-styled Chain-of-Thought Reasoning in LLMs
- **Link:** [arxiv 2601.03559](https://arxiv.org/abs/2601.03559)
- **Authors:** Cao, Lin, Gu, Luo, Ma (2026)
- **Key contribution:** Reformulates CoT reasoning as iterative denoising using a sliding-window mechanism over reasoning steps.
- **Relevance:** For ARC-AGI, could help your model learn multi-step reasoning as a diffusion process over abstract reasoning states.

### LogicDiff: Logic-Guided Denoising for MDLM Zero-Shot Reasoning
- **Link:** [arxiv 2603.26771](https://arxiv.org/abs/2603.26771)
- **Authors:** Aman (2026)
- **Key contribution:** Replaces confidence-based unmasking with logic-role-guided unmasking using a lightweight classification head (4.2M params). Systematic improvement on reasoning tasks.
- **Relevance:** ⭐ **High impact for RDLM.** The core insight — that confidence-based sampling defers high-entropy logical tokens — likely applies to ARC grid-prediction tasks as well.

---

## 6. Scaling Laws & Comparisons

### Parallelism and Generation Order in MDLMs
- **Link:** [arxiv 2601.15593](https://arxiv.org/abs/2601.15593)
- **Authors:** Zhong, Gu, Zang, Li, Ding, Jia, Shen, Lan, Zhu, Liu, Zhou, Liu, Yu, Luo, Qi, Yan, Zhao (2026)
- **Key contribution:** Large-scale study of 8 MDLMs up to 100B parameters across 58 benchmarks. Shows current MDLMs still exhibit strong left-to-right bias and limited parallelism. Only up to ~25% of tokens can be predicted in parallel before quality degrades.
- **Relevance:** Important context for setting realistic expectations about parallel generation in your model.

### Autoregressive vs. Masked Diffusion Language Models: A Controlled Comparison
- **Link:** [arxiv 2603.22075](https://arxiv.org/abs/2603.22075)
- **Authors:** Vicentino (2026)
- **Key contribution:** Controlled comparison showing the pros/cons of each paradigm.
- **Relevance:** Provides scientific context for your architectural choices.

---

## 7. Recursive/Iterative Architectures (TRM Family)

### Less is More: Recursive Reasoning with Tiny Networks (TRM)
- **Link:** [arxiv 2510.04871](https://arxiv.org/abs/2510.04871)
- **Authors:** Jolicoeur-Martineau (2025)
- **Key contribution:** Proposes TRM — two small neural networks recursing at different frequencies. Beats LLMs on Sudoku, Maze, and ARC-AGI with only 27M parameters trained on ~1000 examples.
- **Relevance:** ⭐ **The foundational paper for your backbone architecture.** Your `trm.py` implements this. Essential reading for understanding your own model's design.

### Tiny Recursive Reasoning with Mamba-2 Attention Hybrid
- **Link:** [arxiv 2602.12078](https://arxiv.org/abs/2602.12078)
- **Authors:** Wang, Reid (2026)
- **Key contribution:** Explores replacing attention with Mamba-2's state space model in the TRM architecture. Finds Mamba-2's inherent iterative refinement complements recursive reasoning.
- **Relevance:** ⭐ **High impact for RDLM.** Your TinyBlock is attention-based — Mamba-2 could be a more efficient alternative that naturally supports the recursive refinement paradigm.

### Tiny Autoregressive Recursive Models
- **Link:** [arxiv 2603.08082](https://arxiv.org/abs/2603.08082)
- **Authors:** Rauba, Fanconi, van der Schaar (2026)
- **Key contribution:** Adapts TRM's refinement mechanism to autoregressive models. Shows the refinement mechanism generalizes beyond the original formulation.
- **Relevance:** Validates that the recursive refinement idea is broad and powerful — your diffusion+recursion combo is well-motivated.

### Vision Tiny Recursion Model (ViTRM)
- **Link:** [arxiv 2603.19503](https://arxiv.org/abs/2603.19503)
- **Authors:** Akazan, Koroko, Mbingui, Arinloye, Fifen, Bandolo (2026)
- **Key contribution:** Adapts TRM to image classification. Shows the recursive state refinement paradigm generalizes beyond reasoning tasks.
- **Relevance:** Demonstrates TRM's versatility — relevant for your ARC grid tasks which have a visual/structured component.

### Tab-TRM: Tiny Recursive Model for Insurance Pricing
- **Link:** [arxiv 2601.07675](https://arxiv.org/abs/2601.07675)
- **Authors:** Padayachy, Richman, Wüthrich (2026)
- **Key contribution:** Adapts TRM to tabular data with two learnable latent tokens (answer + reasoning state) iteratively refined.
- **Relevance:** Their adaptation of TRM to structured data is analogous to your adaptation for ARC grids.

### Test-time Adaptation of Tiny Recursive Models
- **Link:** [arxiv 2511.02886](https://arxiv.org/abs/2511.02886)
- **Authors:** McGovern (2025)
- **Key contribution:** Methods for adapting TRM at test time without retraining.
- **Relevance:** Could be applied to your ARC-AGI evaluation for better generalization to novel tasks.

---

## 8. ARC-AGI Benchmarks & Approaches

### ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems
- **Link:** [arxiv 2505.11831](https://arxiv.org/abs/2505.11831)
- **Authors:** Chollet, Knoop, Kamradt, Landers, Pinkard (2025)
- **Key contribution:** Introduces ARC-AGI-2, a harder benchmark with higher cognitive complexity. Most contemporary systems score below 4%.
- **Relevance:** After mastering ARC-AGI-1, ARC-AGI-2 is the next target for your model.

### Executable World Models for ARC-AGI-3 in the Era of Coding Agents
- **Link:** [arxiv 2605.05138](https://arxiv.org/abs/2605.05138)
- **Authors:** Rodionov (2026)
- **Key contribution:** Evaluates a coding-agent system for ARC-AGI-3 that maintains an executable Python world model, verifies against observations, and refactors toward simpler abstractions.
- **Relevance:** Interesting complementary approach to pure neural methods — could be combined with your diffusion model.

### ARC-AGI Without Pretraining (CompressARC)
- **Link:** [arxiv 2512.06104](https://arxiv.org/abs/2512.06104)
- **Authors:** Liao, Gu (2025)
- **Key contribution:** 76K parameter model without any pretraining solves 20% of ARC-AGI-1 evaluation puzzles by minimizing description length (MDL) purely at inference time.
- **Relevance:** ⭐ **Fascinating approach.** The MDL principle could potentially complement your diffusion model's confidence-based selection — the lowest description length solution is often the correct one.

### The ARC of Progress towards AGI: A Living Survey
- **Link:** [arxiv 2603.13372](https://arxiv.org/abs/2603.13372)
- **Authors:** Vahdati, Aioanei, Suresh, Lehmann (2026)
- **Key contribution:** Comprehensive survey of ARC-AGI approaches, including neural, symbolic, and hybrid methods.
- **Relevance:** Good reference for positioning your work and finding complementary techniques.

### ARC-TGI: Human-Validated Task Generators for ARC-AGI
- **Link:** [arxiv 2603.05099](https://arxiv.org/abs/2603.05099)
- **Authors:** Lehmann et al. (2026)
- **Key contribution:** Task generators with reasoning chain templates for augmented ARC-AGI training data.
- **Relevance:** Could help you generate more training data for your diffusion model.

---

## 9. Guidance & Control for Discrete Diffusion

### Simple Guidance Mechanisms for Discrete Diffusion Models
- **Link:** [arxiv 2412.10193](https://arxiv.org/abs/2412.10193)
- **Authors:** Schiff, Sahoo, Phung, Wang, Boshar, Dalla-torre, de Almeida, Rush, Pierrot, Kuleshov (2024)
- **Key contribution:** Derives classifier-free and classifier-based guidance for discrete diffusion. Shows uniform-noise models are more guidable.
- **Relevance:** Could enable controlled generation in your ARC model (e.g., guide toward certain grid properties).

### Sub-GoL: Discrete Diffusion Modeling by Estimating Ratios (D3PM / Score Entropy)
- **Link:** [arxiv 2310.16834](https://arxiv.org/abs/2310.16834)
- **Authors:** Lou, Meng, Ermon (2023)
- **Key contribution:** Proposes score entropy — a natural extension of score matching to discrete spaces. Bridges the gap between continuous and discrete diffusion theory.
- **Relevance:** Theoretical foundation for understanding discrete diffusion better.

### Unified Discrete Diffusion for Categorical Data (UDD)
- **Link:** [arxiv 2402.03701](https://arxiv.org/abs/2402.03701)
- **Authors:** Zhao, Ding, Yu, Akoglu (2024)
- **Key contribution:** Unifies discrete-time and continuous-time discrete diffusion frameworks.
- **Relevance:** Provides theoretical clarity on which discrete diffusion formulation to use.

### A Reparameterized Discrete Diffusion Model for Text Generation
- **Link:** [arxiv 2302.05737](https://arxiv.org/abs/2302.05737)
- **Authors:** Zheng, Yuan, Yu, Kong (2023)
- **Key contribution:** Reparameterization trick for discrete diffusion to reduce variance.
- **Relevance:** Training stability improvement that could apply to your model.

---

## 10. Discrete Diffusion Foundations

| Year | Paper | Link | Key Idea |
|------|-------|------|----------|
| 2022 | DiffusionBERT | [2211.15029](https://arxiv.org/abs/2211.15029) | BERT + diffusion noise schedule |
| 2023 | D3PM / Score Entropy | [2310.16834](https://arxiv.org/abs/2310.16834) | Score matching for discrete data |
| 2024 | MDLM | [2406.07524](https://arxiv.org/abs/2406.07524) | Simple masked discrete diffusion works |
| 2024 | UDD | [2402.03701](https://arxiv.org/abs/2402.03701) | Unified discrete diffusion framework |
| 2024 | Discrete Flow Matching | [2407.15595](https://arxiv.org/abs/2407.15595) | Flow matching on discrete spaces |
| 2025 | dLLM | [2602.22661](https://arxiv.org/abs/2602.22661) | Standardized diffusion LM components |
| 2025 | TRM | [2510.04871](https://arxiv.org/abs/2510.04871) | Tiny recursive reasoning networks |
| 2026 | Scaling Beyond MDLM | [2602.15014](https://arxiv.org/abs/2602.15014) | Scaling laws for discrete diffusion |

---

## 11. Quick Wins for RDLM

These are directly actionable improvements, ordered by estimated effort/impact ratio:

### Easy integrations (implement in days):
1. **Dependency-Oriented Sampler (DOS)** — Replace confidence-based top-k unmasking with their sequence-level dependency-aware approach. Training-free. → `_sample_with_scores` in `arc_model.py` and `sample` in `diffusion_lm.py`
2. **Temporal Self-Consistency Voting** — Track predictions across denoising steps and vote on final tokens. Training-free. → `_sample_with_scores` in `arc_model.py`
3. **Dilated Unmasking Scheduler (DUS)** — Better parallel unmasking within blocks. Training-free. → `get_num_transfer_tokens` / block sampling logic

### Medium integrations (implement in weeks):
4. **MDPO-style training** — Fix training-inference mismatch by training with structured unmasking schedules instead of random masking. → `_apply_mask` in `diffusion_lm.py` and training loop in `train_arc.py`
5. **Soft-Masked Diffusion** — Replace binary mask-or-reveal with soft mixing weighted by confidence. → `_apply_mask` / backbone token embedding
6. **Where-to-Unmask aux head** — Learn unmasking order via lightweight prediction head. → Add head to backbone, train with auxiliary loss
7. **LogicDiff/Role-guided unmasking** — Classify token roles and prioritize accordingly. → Add classification head, guide unmasking order

### High impact (longer-term):
8. **RL for diffusion trajectory (d1)** — Apply online RL to optimize the denoising process end-to-end. → Requires RL infrastructure
9. **UnMaskFork (MCTS test-time search)** — Branching search over unmasking decisions. → Can use your existing ensemble infrastructure
10. **Self-Rewarding SMC for MDLMs** — Particle filtering over denoising trajectories. → Sophisticated but principled
11. **Mamba-2 in TRM** — Replace attention with state space model. → Could improve efficiency of recursive refinement

### ARC-AGI specific:
12. **CompressARC's MDL principle** — Use description length minimization alongside diffusion confidence for better ARC candidate selection.
13. **ARC-TGI augmented data** — Generate more training tasks to improve coverage.
14. **Test-time adaptation of TRM** — Adapt your model to novel ARC tasks at inference time.

---

## Reading Quick-Start

If you only read **5 papers**, start here:

1. **[TRM (2510.04871)](https://arxiv.org/abs/2510.04871)** — Understand your own backbone architecture's origins
2. **[MDLM (2406.07524)](https://arxiv.org/abs/2406.07524)** — The diffusion framework you're building on
3. **[dLLM (2602.22661)](https://arxiv.org/abs/2602.22661)** — Unified reference implementation, directly cited in your code
4. **[MDPO (2508.13148)](https://arxiv.org/abs/2508.13148)** — Fixes the train-inference divide that plagues MDLMs
5. **[Time Is a Feature (2508.09138)](https://arxiv.org/abs/2508.09138)** — Temporal oscillation insight + practical mitigation

### Most underrated:
- **[Soft-Masked (2510.17206)](https://arxiv.org/abs/2510.17206)** — Simple architectural change, could significantly improve your generation quality
- **[Scaling Beyond MDLM (2602.15014)](https://arxiv.org/abs/2602.15014)** — Practical scaling insights + better loss weighting
- **[CompressARC (2512.06104)](https://arxiv.org/abs/2512.06104)** — Mind-blowing result: 76K params, no pretraining, 20% on ARC-AGI via MDL
