# Ranked Research Bibliography for RDLM

Total gathered papers: **126**.

## Ranking formula

Keyword score over title/abstract/query metadata: direct code relevance + topic coverage + recency + citation impact + small-model/reasoning bonuses - large-scale penalties.

## Theme counts

- **Recursive / iterative reasoning**: 73
- **Small-model reasoning**: 42
- **Diffusion / denoising LM**: 7
- **ARC / program / symbolic reasoning**: 3
- **Curriculum / synthetic data / RL**: 1

## Top papers for deep synthesis

### 1. One Step Forward and K Steps Back: Better Reasoning with Denoising Recursion Models
- **Score**: 51.1
- **Theme**: Recursive / iterative reasoning
- **Authors**: Chris Cameron et al.
- **Year / venue**: 2026 / n/a
- **URL**: https://www.semanticscholar.org/paper/b433acdd80aed14800400721a2e9edcaf1dc8928
- **Topics**: recursive reasoning, small models, diffusion LM, ARC/abstract reasoning, curriculum/synthetic/RL
- **Factual summary seed**: Looped transformers scale computational depth without increasing parameter count by repeatedly applying a shared transformer block and can be used for iterative refinement, where each loop rewrites a full fixed-size prediction in parallel. On difficult problems, such as those th…
- **Implementation relevance**: Directly relevant to ARC evaluation, object/grid priors, or candidate verification.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 2. Symbol-Equivariant Recurrent Reasoning Models
- **Score**: 50.1
- **Theme**: Recursive / iterative reasoning
- **Authors**: Richard Freinschlag et al.
- **Year / venue**: 2026 / ArXiv.org
- **URL**: https://openalex.org/W7133364588
- **Topics**: ARC/abstract reasoning, recursive reasoning, small models
- **Factual summary seed**: Reasoning problems such as Sudoku and ARC-AGI remain challenging for neural networks. The structured problem solving architecture family of Recurrent Reasoning Models (RRMs), including Hierarchical Reasoning Model (HRM) and Tiny Recursive Model (TRM), offer a compact alternative…
- **Implementation relevance**: Directly relevant to ARC evaluation, object/grid priors, or candidate verification.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 3. ATHENA: A Cognitive Architecture for Persistent Execution and Skill Formation in Autonomous Agents
- **Score**: 50.1
- **Theme**: ARC / program / symbolic reasoning
- **Authors**: Hafizur Rahman
- **Year / venue**: 2026 / Zenodo (CERN European Organization for Nuclear Research)
- **URL**: https://openalex.org/W7124259583
- **Topics**: ARC/abstract reasoning, memory/state-space/sparse, neuro-symbolic/programs, recursive reasoning, small models
- **Factual summary seed**: We present ATHENA (Adaptive Tiered Hybrid Execution and Neuro-Symbolic Architecture), a cognitive architecture for autonomous agents that compiles neural deliberation into persistent executable knowledge. ATHENA integrates neural and symbolic components in a three-tier hierarchy…
- **Implementation relevance**: Directly relevant to ARC evaluation, object/grid priors, or candidate verification.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 4. Closed-Loop Transformers: Autoregressive Modeling as Iterative Latent Equilibrium
- **Score**: 48.73
- **Theme**: Recursive / iterative reasoning
- **Authors**: Akbar Anbar Jafari, G. Anbarjafari
- **Year / venue**: 2025 / arXiv.org
- **URL**: https://www.semanticscholar.org/paper/cfefe7a5df5b04883ef63bfb7dd675682d6b0085
- **Topics**: recursive reasoning, diffusion LM, test-time compute, memory/state-space/sparse, curriculum/synthetic/RL
- **Factual summary seed**: Contemporary autoregressive transformers operate in open loop: each hidden state is computed in a single forward pass and never revised, causing errors to propagate uncorrected through the sequence. We identify this open-loop bottleneck as a fundamental architectural limitation…
- **Implementation relevance**: Relevant to RDLM masking schedules, denoising objectives, and decoding trajectories.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 5. Hierarchical Reasoning Model
- **Score**: 47.8
- **Theme**: ARC / program / symbolic reasoning
- **Authors**: Guan Wang et al.
- **Year / venue**: 2025 / arXiv
- **URL**: http://arxiv.org/abs/2506.21734v3
- **Topics**: recursive reasoning, latent/CoT reasoning, ARC/abstract reasoning, curriculum/synthetic/RL
- **Factual summary seed**: Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI. Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requ…
- **Implementation relevance**: Directly relevant to ARC evaluation, object/grid priors, or candidate verification.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 6. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **Score**: 41.2
- **Theme**: Recursive / iterative reasoning
- **Authors**: Albert Gu, Tri Dao
- **Year / venue**: 2023 / arXiv (Cornell University)
- **URL**: https://openalex.org/W4389326242
- **Topics**: memory/state-space/sparse, recursive reasoning, small language models, small models
- **Factual summary seed**: Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models,…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 7. rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking
- **Score**: 38.16
- **Theme**: Small-model reasoning
- **Authors**: Xinyu Guan et al.
- **Year / venue**: 2025 / arXiv (Cornell University)
- **URL**: https://openalex.org/W4406231568
- **Topics**: small models, test-time compute, latent/CoT reasoning, curriculum/synthetic/RL, small language models
- **Factual summary seed**: We present rStar-Math to demonstrate that small language models (SLMs) can rival or even surpass the math reasoning capability of OpenAI o1, without distillation from superior models. rStar-Math achieves this by exercising "deep thinking" through Monte Carlo Tree Search (MCTS),…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 8. Multimodal Latent Reasoning via Hierarchical Visual Cues Injection
- **Score**: 36.6
- **Theme**: Recursive / iterative reasoning
- **Authors**: Yiming Zhang et al.
- **Year / venue**: 2026 / arXiv.org
- **URL**: https://www.semanticscholar.org/paper/30e7b2338f9fada55addb68e870803975da2bcc2
- **Topics**: recursive reasoning, small models, test-time compute, latent/CoT reasoning
- **Factual summary seed**: The advancement of multimodal large language models (MLLMs) has enabled impressive perception capabilities. However, their reasoning process often remains a"fast thinking"paradigm, reliant on end-to-end generation or explicit, language-centric chains of thought (CoT), which can…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 9. Virtual Parameter Sharpening: Dynamic Low-Rank Perturbations for Inference-Time Reasoning Enhancement
- **Score**: 34.3
- **Theme**: Recursive / iterative reasoning
- **Authors**: Saba Kublashvili
- **Year / venue**: 2025 / arXiv.org
- **URL**: https://www.semanticscholar.org/paper/d1dafc2dba42c7702f31083b4fe3fcc32d762537
- **Topics**: recursive reasoning, small models, test-time compute
- **Factual summary seed**: I introduce Virtual Parameter Sharpening (VPS), an inference-time technique that augments frozen transformer linear layers with dynamic, activation-conditioned low-rank perturbations. Unlike parameter-efficient fine-tuning methods such as LoRA, which learn static low-rank adapte…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 10. Artificial Intelligence for Materials Discovery, Development, and Optimization
- **Score**: 33.12
- **Theme**: Recursive / iterative reasoning
- **Authors**: Benediktus Madika et al.
- **Year / venue**: 2025 / ACS Nano
- **URL**: https://openalex.org/W4412654790
- **Topics**: recursive reasoning, curriculum/synthetic/RL, mechanistic interpretability
- **Factual summary seed**: This review highlights the recent transformative impact of artificial intelligence (AI), machine learning (ML), and deep learning (DL) on materials science, emphasizing their applications in materials discovery, development, and optimization. AI-driven methods have revolutionize…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 11. Dynamical Systems Theory Behind a Hierarchical Reasoning Model
- **Score**: 33.1
- **Theme**: ARC / program / symbolic reasoning
- **Authors**: Vasiliy A. Es'kin, Mikhail E. Smorkalov
- **Year / venue**: 2026 / arXiv
- **URL**: http://arxiv.org/abs/2603.22871v1
- **Topics**: ARC/abstract reasoning, recursive reasoning, small models
- **Factual summary seed**: Current large language models (LLMs) primarily rely on linear sequence generation and massive parameter counts, yet they severely struggle with complex algorithmic reasoning. While recent reasoning architectures, such as the Hierarchical Reasoning Model (HRM) and Tiny Recursive…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 12. A Survey of Large Language Models
- **Score**: 32.6
- **Theme**: Small-model reasoning
- **Authors**: Wayne Xin Zhao et al.
- **Year / venue**: 2026 / Frontiers of Computer Science
- **URL**: https://openalex.org/W4362515116
- **Topics**: curriculum/synthetic/RL, recursive reasoning, small language models, small models
- **Factual summary seed**: Abstract The rapid evolution of large language models (LLMs) has driven a transformative shift in artificial intelligence (AI), reshaping both research paradigms and practical applications. Distinguished from their predecessors by unprecedented scale and advanced capabilities, L…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 13. Beyond Test-Time Compute Strategies: Advocating Energy-per-Token in LLM Inference
- **Score**: 32.4
- **Theme**: Small-model reasoning
- **Authors**: Pascal Wilhelm, Thorsten Wittkopp, Odej Kao
- **Year / venue**: 2025 / n/a
- **URL**: https://openalex.org/W4409047589
- **Topics**: small models, test-time compute, latent/CoT reasoning, curriculum/synthetic/RL, small language models
- **Factual summary seed**: Large Language Models (LLMs) demonstrate exceptional performance across diverse tasks but come with substantial energy and computational costs, particularly in request-heavy scenarios. In many real-world applications, the full scale and capabilities of LLMs are often unnecessary…
- **Implementation relevance**: Relevant to low-risk inference-time reranking, voting, and extra-compute allocation.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 14. Neural-Symbolic Collaborative Distillation: Advancing Small Language Models for Complex Reasoning Tasks
- **Score**: 32.11
- **Theme**: Small-model reasoning
- **Authors**: Huanxuan Liao et al.
- **Year / venue**: 2025 / Proceedings of the AAAI Conference on Artificial Intelligence
- **URL**: https://openalex.org/W4409348057
- **Topics**: neuro-symbolic/programs, small language models, small models
- **Factual summary seed**: In this paper, we propose Neural-Symbolic Collaborative Distillation (NesyCD), a novel knowledge distillation method for learning the complex reasoning abilities of Large Language Models (LLMs, e.g., \textgreater 13B). We argue that complex reasoning tasks are difficult for Smal…
- **Implementation relevance**: Directly relevant to ARC evaluation, object/grid priors, or candidate verification.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 15. Toward expert-level medical question answering with large language models
- **Score**: 31.8
- **Theme**: Recursive / iterative reasoning
- **Authors**: K. K. Singhal et al.
- **Year / venue**: 2025 / Nature Medicine
- **URL**: https://openalex.org/W4406152279
- **Topics**: curriculum/synthetic/RL, recursive reasoning
- **Factual summary seed**: Large language models (LLMs) have shown promise in medical question answering, with Med-PaLM being the first to exceed a 'passing' score in United States Medical Licensing Examination style questions. However, challenges remain in long-form medical question answering and handlin…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 16. PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation
- **Score**: 30.81
- **Theme**: Recursive / iterative reasoning
- **Authors**: Qiyao Xue et al.
- **Year / venue**: 2024 / Computer Vision and Pattern Recognition
- **URL**: https://www.semanticscholar.org/paper/015b1f127b6c31654e3597b75876eed8e445d866
- **Topics**: diffusion LM, curriculum/synthetic/RL, recursive reasoning
- **Factual summary seed**: Text-to-video (T2V) generation has been recently enabled by transformer-based diffusion models, but current T2V models lack capabilities in adhering to the real-world common knowledge and physical rules, due to their limited understanding of physical realism and deficiency in te…
- **Implementation relevance**: Relevant to RDLM masking schedules, denoising objectives, and decoding trajectories.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 17. Simplified and Generalized Masked Diffusion for Discrete Data
- **Score**: 30.81
- **Theme**: Diffusion / denoising LM
- **Authors**: Jiaxin Shi et al.
- **Year / venue**: 2024 / arXiv (Cornell University)
- **URL**: https://openalex.org/W4399455367
- **Topics**: diffusion LM, diffusion text
- **Factual summary seed**: Masked (or absorbing) diffusion is actively explored as an alternative to autoregressive models for generative modeling of discrete data. However, existing work in this area has been hindered by unnecessarily complex model formulations and unclear relationships between different…
- **Implementation relevance**: Relevant to RDLM masking schedules, denoising objectives, and decoding trajectories.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 18. A Comprehensive Evaluation of Transformer-Based Question Answering Models and RAG-Enhanced Design
- **Score**: 30.8
- **Theme**: Recursive / iterative reasoning
- **Authors**: Zichen Zhang et al.
- **Year / venue**: 2025 / arXiv.org
- **URL**: https://www.semanticscholar.org/paper/8142b1dbd9163f9e5166da2d25c1c9c1f90c74cb
- **Topics**: recursive reasoning, small models, curriculum/synthetic/RL, mechanistic interpretability
- **Factual summary seed**: Transformer-based models have advanced the field of question answering, but multi-hop reasoning, where answers require combining evidence across multiple passages, remains difficult. This paper presents a comprehensive evaluation of retrieval strategies for multi-hop question an…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 19. Revisiting Large Language Models as Zero-shot Relation Extractors
- **Score**: 30.57
- **Theme**: Recursive / iterative reasoning
- **Authors**: Guozheng Li, Peng Wang, Wenjun Ke
- **Year / venue**: 2023 / n/a
- **URL**: https://openalex.org/W4389518985
- **Topics**: recursive reasoning, small models, latent/CoT reasoning, curriculum/synthetic/RL, small language models
- **Factual summary seed**: Relation extraction (RE) consistently involves a certain degree of labeled or unlabeled data even if under zero-shot setting. Recent studies have shown that large language models (LLMs) transfer well to new tasks out-of-the-box simply given a natural language prompt, which provi…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 20. Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing
- **Score**: 30.4
- **Theme**: Small-model reasoning
- **Authors**: Pengfei Liu et al.
- **Year / venue**: 2022 / ACM Computing Surveys
- **URL**: https://openalex.org/W3185341429
- **Topics**: curriculum/synthetic/RL, small language models
- **Factual summary seed**: This article surveys and organizes research works in a new paradigm in natural language processing, which we dub “prompt-based learning.” Unlike traditional supervised learning, which trains a model to take in an input x and predict an output y as P ( y|x ), prompt-based learnin…
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 21. Image Super-Resolution Via Iterative Refinement
- **Score**: 30.4
- **Theme**: Recursive / iterative reasoning
- **Authors**: Chitwan Saharia et al.
- **Year / venue**: 2022 / IEEE Transactions on Pattern Analysis and Machine Intelligence
- **URL**: https://openalex.org/W3155072588
- **Topics**: recursive reasoning, diffusion LM
- **Factual summary seed**: We present SR3, an approach to image Super-Resolution via Repeated Refinement. SR3 adapts denoising diffusion probabilistic models (Ho et al. 2020), (Sohl-Dickstein et al. 2015) to image-to-image translation, and performs super-resolution through a stochastic iterative denoising…
- **Implementation relevance**: Relevant to RDLM masking schedules, denoising objectives, and decoding trajectories.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 22. MAGDi: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models
- **Score**: 30.4
- **Theme**: Small-model reasoning
- **Authors**: Justin Chih-Yao Chen et al.
- **Year / venue**: 2024 / arXiv (Cornell University)
- **URL**: https://openalex.org/W4391556796
- **Topics**: small models, test-time compute, small language models
- **Factual summary seed**: Multi-agent interactions between Large Language Model (LLM) agents have shown major improvements on diverse reasoning tasks. However, these involve long generations from multiple models across several rounds, making them expensive. Moreover, these multi-agent approaches fail to…
- **Implementation relevance**: Relevant to low-risk inference-time reranking, voting, and extra-compute allocation.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 23. Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning
- **Score**: 30.31
- **Theme**: Small-model reasoning
- **Authors**: Amrith Setlur et al.
- **Year / venue**: 2024 / arXiv (Cornell University)
- **URL**: https://openalex.org/W4403365357
- **Topics**: small models, test-time compute, latent/CoT reasoning, curriculum/synthetic/RL, small language models
- **Factual summary seed**: A promising approach for improving reasoning in large language models is to use process reward models (PRMs). PRMs provide feedback at each step of a multi-step reasoning trace, potentially improving credit assignment over outcome reward models (ORMs) that only provide feedback…
- **Implementation relevance**: Relevant to low-risk inference-time reranking, voting, and extra-compute allocation.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 24. Can Small Language Models Help Large Language Models Reason Better?: LM-Guided Chain-of-Thought
- **Score**: 29.9
- **Theme**: Small-model reasoning
- **Authors**: Jooyoung Lee et al.
- **Year / venue**: 2024 / International Conference on Language Resources and Evaluation
- **URL**: https://www.semanticscholar.org/paper/f19ddba513dfeca571a6e2ba7542a63677cd5e3b
- **Topics**: curriculum/synthetic/RL, latent/CoT reasoning, small language models, small models
- **Factual summary seed**: We introduce a novel framework, LM-Guided CoT, that leverages a lightweight (i.e., <1B) language model (LM) for guiding a black-box large (i.e., >10B) LM in reasoning tasks. Specifically, the lightweight LM first generates a rationale for each input instance. The Frozen large LM…
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 25. Towards Expert-Level Medical Question Answering with Large Language Models
- **Score**: 29.77
- **Theme**: Recursive / iterative reasoning
- **Authors**: Karan Singhal et al.
- **Year / venue**: 2023 / arXiv (Cornell University)
- **URL**: https://openalex.org/W4377009978
- **Topics**: curriculum/synthetic/RL, recursive reasoning
- **Factual summary seed**: Recent artificial intelligence (AI) systems have reached milestones in "grand challenges" ranging from Go to protein-folding. The capability to retrieve medical knowledge, reason over it, and answer medical questions comparably to physicians has long been viewed as one such gran…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 26. Diffutron: A Masked Diffusion Language Model for Turkish Language
- **Score**: 29.6
- **Theme**: Diffusion / denoising LM
- **Authors**: Şuayp Talha Kocabay, Talha Rüzgar Akkuş
- **Year / venue**: 2026 / arXiv (Cornell University)
- **URL**: https://openalex.org/W7140346153
- **Topics**: small models, diffusion LM, neuro-symbolic/programs, diffusion text
- **Factual summary seed**: Masked Diffusion Language Models (MDLMs) have emerged as a compelling non-autoregressive alternative to standard large language models; however, their application to morphologically rich languages remains limited. In this paper, we introduce $\textit{Diffutron}$, a masked diffus…
- **Implementation relevance**: Relevant to RDLM masking schedules, denoising objectives, and decoding trajectories.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 27. A survey on large language model based autonomous agents
- **Score**: 29.5
- **Theme**: Recursive / iterative reasoning
- **Authors**: Lei Wang et al.
- **Year / venue**: 2024 / Frontiers of Computer Science
- **URL**: https://openalex.org/W4393065402
- **Topics**: recursive reasoning
- **Factual summary seed**: Abstract Autonomous agents have long been a research focus in academic and industry communities. Previous research often focuses on training agents with limited knowledge within isolated environments, which diverges significantly from human learning processes, and makes the agen…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 28. Object Detection in 20 Years: A Survey
- **Score**: 29.5
- **Theme**: Recursive / iterative reasoning
- **Authors**: Zhengxia Zou et al.
- **Year / venue**: 2019 / arXiv (Cornell University)
- **URL**: https://openalex.org/W2944165510
- **Topics**: neuro-symbolic/programs, curriculum/synthetic/RL, recursive reasoning
- **Factual summary seed**: Object detection, as of one the most fundamental and challenging problems in computer vision, has received great attention in recent years. Over the past two decades, we have seen a rapid technological evolution of object detection and its profound impact on the entire computer…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 29. Embodied Reasoning with Self-Feedback
- **Score**: 29.43
- **Theme**: Recursive / iterative reasoning
- **Authors**: Pranav Kak, Sushma Jain
- **Year / venue**: 2024 / IEEE International Conference on Electronics, Computing and Communication Technologies
- **URL**: https://www.semanticscholar.org/paper/4c2426f47d90839301899a28117c71fae3179b68
- **Topics**: ARC/abstract reasoning, recursive reasoning
- **Factual summary seed**: Large Language Models (LLMs) based on Transformer architecture have achieved remarkable success in Natural Language Processing (NLP). However, translating open-ended instructions into granular action plans for robotic systems remains a challenge. This paper proposes a self-super…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 30. Graph of Thoughts: Solving Elaborate Problems with Large Language Models
- **Score**: 29.27
- **Theme**: Recursive / iterative reasoning
- **Authors**: Maciej Besta et al.
- **Year / venue**: 2024 / Proceedings of the AAAI Conference on Artificial Intelligence
- **URL**: https://openalex.org/W4393160302
- **Topics**: recursive reasoning
- **Factual summary seed**: We introduce Graph of Thoughts (GoT): a framework that advances prompting capabilities in large language models (LLMs) beyond those offered by paradigms such as Chain-of-Thought or Tree of Thoughts (ToT). The key idea and primary advantage of GoT is the ability to model the info…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 31. Multi-Hop Knowledge Graph Reasoning with Reward Shaping
- **Score**: 29.23
- **Theme**: Recursive / iterative reasoning
- **Authors**: Xi Lin, Richard Socher, Caiming Xiong
- **Year / venue**: 2018 / n/a
- **URL**: https://openalex.org/W2889344053
- **Topics**: test-time compute, curriculum/synthetic/RL, recursive reasoning
- **Factual summary seed**: Multi-hop reasoning is an effective approach for query answering (QA) over incomplete knowledge graphs (KGs). The problem can be formulated in a reinforcement learning (RL) setup, where a policy-based agent sequentially extends its inference path until it reaches a target. Howev…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 32. Knowledge-Augmented Reasoning Distillation for Small Language Models in Knowledge-Intensive Tasks
- **Score**: 29.23
- **Theme**: Small-model reasoning
- **Authors**: Minki Kang et al.
- **Year / venue**: 2023 / arXiv (Cornell University)
- **URL**: https://openalex.org/W4378942311
- **Topics**: curriculum/synthetic/RL, small language models, small models, test-time compute
- **Factual summary seed**: Large Language Models (LLMs) have shown promising performance in knowledge-intensive reasoning tasks that require a compound understanding of knowledge. However, deployment of the LLMs in real-world applications can be challenging due to their high computational requirements and…
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 33. MizAR 60 for Mizar 50
- **Score**: 29.2
- **Theme**: Recursive / iterative reasoning
- **Authors**: Jakubův, Jan et al.
- **Year / venue**: 2023 / DROPS (Schloss Dagstuhl – Leibniz Center for Informatics)
- **URL**: https://openalex.org/W4385245566
- **Topics**: curriculum/synthetic/RL, recursive reasoning
- **Factual summary seed**: As a present to Mizar on its 50th anniversary, we develop an AI/TP system that automatically proves about 60% of the Mizar theorems in the hammer setting. We also automatically prove 75% of the Mizar theorems when the automated provers are helped by using only the premises used…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 34. One-Shot Autoregressive Generation of Combinatorial Optimization Solutions Based on the Large Language Model Architecture and Learning Algorithms
- **Score**: 29.2
- **Theme**: Recursive / iterative reasoning
- **Authors**: Bishad Ghimire, Ausif Mahmood, Khaled Elleithy
- **Year / venue**: 2025 / AI
- **URL**: https://openalex.org/W4408901775
- **Topics**: recursive reasoning, diffusion LM, curriculum/synthetic/RL
- **Factual summary seed**: Large Language Models (LLMs) have immensely advanced the field of Artificial Intelligence (AI), with recent models being able to perform chain-of-thought reasoning and solve complex mathematical problems, ranging from theorem proving to ones involving advanced calculus. The succ…
- **Implementation relevance**: Relevant to RDLM masking schedules, denoising objectives, and decoding trajectories.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 35. Distilling Reasoning Capabilities into Smaller Language Models
- **Score**: 29.1
- **Theme**: Small-model reasoning
- **Authors**: Kumar Shridhar, Alessandro Stolfo, Mrinmaya Sachan
- **Year / venue**: 2023 / arXiv (Cornell University)
- **URL**: https://openalex.org/W4385571831
- **Topics**: latent/CoT reasoning, small language models, small models
- **Factual summary seed**: Step-by-step reasoning approaches like chain of thought (CoT) have proved to be very effective in inducing reasoning capabilities in large language models. However, the success of the CoT approach is fundamentally tied to the model size, and billion parameter-scale models are of…
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 36. A Metaverse: Taxonomy, Components, Applications, and Open Challenges
- **Score**: 28.9
- **Theme**: Small-model reasoning
- **Authors**: Sangmin Park, Young‐Gab Kim
- **Year / venue**: 2022 / IEEE Access
- **URL**: https://openalex.org/W4206484811
- **Topics**: neuro-symbolic/programs, recursive reasoning, small language models
- **Factual summary seed**: Unlike previous studies on the Metaverse based on Second Life, the current Metaverse is based on the social value of Generation Z that online and offline selves are not different. With the technological development of deep learning-based high-precision recognition models and nat…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 37. Weight-Tied Adaptive Recursive Vision–Language–Action Transformer for Efficient Multimodal Robotic Control
- **Score**: 28.6
- **Theme**: Recursive / iterative reasoning
- **Authors**: Howaida Allam, Inam Ullah Khan
- **Year / venue**: 2026 / Journal of Smart Algorithms and Applications (JSAA)
- **URL**: https://www.semanticscholar.org/paper/d12bf0d70d5c748bf5354542efc098db26b24b9f
- **Topics**: recursive reasoning, small models
- **Factual summary seed**: Vision-Language-Action (VLA) models unify perception, language understanding, and control within a single learning framework, enabling robots to execute manipulation tasks specified through natural language and visual observations. Despite recent progress, many existing VLA syst…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 38. A comprehensive survey on pretrained foundation models: a history from BERT to ChatGPT
- **Score**: 28.17
- **Theme**: Recursive / iterative reasoning
- **Authors**: Ce Zhou et al.
- **Year / venue**: 2024 / International Journal of Machine Learning and Cybernetics
- **URL**: https://openalex.org/W4404658388
- **Topics**: recursive reasoning
- **Factual summary seed**: No abstract available.
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 39. Applications of Artificial Intelligence in Transport: An Overview
- **Score**: 28.0
- **Theme**: Recursive / iterative reasoning
- **Authors**: Rusul Abduljabbar et al.
- **Year / venue**: 2019 / Sustainability
- **URL**: https://openalex.org/W2908162093
- **Topics**: small models, neuro-symbolic/programs, curriculum/synthetic/RL, recursive reasoning
- **Factual summary seed**: The rapid pace of developments in Artificial Intelligence (AI) is providing unprecedented opportunities to enhance the performance of different industries and businesses, including the transport sector. The innovations introduced by AI include highly advanced computational metho…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 40. Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning
- **Score**: 27.88
- **Theme**: Small-model reasoning
- **Authors**: Hao Shao et al.
- **Year / venue**: 2024 / Neural Information Processing Systems
- **URL**: https://www.semanticscholar.org/paper/ee8c1a46c90f1261c23479e15c6bed7f67ad8943
- **Topics**: latent/CoT reasoning, mechanistic interpretability, small language models
- **Factual summary seed**: Multi-Modal Large Language Models (MLLMs) have demonstrated impressive performance in various VQA tasks. However, they often lack interpretability and struggle with complex visual inputs, especially when the resolution of the input image is high or when the interested region tha…
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 41. Uncovering Graph Reasoning in Decoder-only Transformers with Circuit Tracing
- **Score**: 27.8
- **Theme**: Recursive / iterative reasoning
- **Authors**: Xinnan Dai et al.
- **Year / venue**: 2025 / arXiv
- **URL**: http://arxiv.org/abs/2509.20336v1
- **Topics**: latent/CoT reasoning, curriculum/synthetic/RL, mechanistic interpretability, recursive reasoning
- **Factual summary seed**: Transformer-based LLMs demonstrate strong performance on graph reasoning tasks, yet their internal mechanisms remain underexplored. To uncover these reasoning process mechanisms in a fundamental and unified view, we set the basic decoder-only transformers and explain them using…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 42. Recurrent Memory Transformer
- **Score**: 27.69
- **Theme**: Recursive / iterative reasoning
- **Authors**: Aydar Bulatov, Yuri Kuratov, Mikhail Burtsev
- **Year / venue**: 2022 / arXiv (Cornell University)
- **URL**: https://openalex.org/W4285595435
- **Topics**: recursive reasoning, memory/state-space/sparse
- **Factual summary seed**: Transformer-based models show their effectiveness across multiple domains and tasks. The self-attention allows to combine information from all sequence elements into context-aware representations. However, global and local information has to be stored mostly in the same element-…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 43. Deep multiagent reinforcement learning: challenges and directions
- **Score**: 27.56
- **Theme**: Curriculum / synthetic data / RL
- **Authors**: Annie Wong et al.
- **Year / venue**: 2022 / Artificial Intelligence Review
- **URL**: https://openalex.org/W4306786778
- **Topics**: small models, curriculum/synthetic/RL, recursive reasoning
- **Factual summary seed**: Abstract This paper surveys the field of deep multiagent reinforcement learning (RL). The combination of deep neural networks with RL has gained increased traction in recent years and is slowly shifting the focus from single-agent to multiagent environments. Dealing with multipl…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 44. Mastering Long-Context Multi-Task Reasoning with Transformers and Recurrent Memory
- **Score**: 27.4
- **Theme**: Recursive / iterative reasoning
- **Authors**: Aleksandr Bulatov, Yuri Kuratov, Mikhail Burtsev
- **Year / venue**: 2024 / Optical Memory and Neural Networks
- **URL**: https://openalex.org/W4406757829
- **Topics**: recursive reasoning, small models, memory/state-space/sparse
- **Factual summary seed**: Recent advancements have significantly improved the skills and performance of language models, but have also increased computational demands due to the increasing number of parameters and the quadratic complexity of the attention mechanism. As context sizes expand into millions…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 45. Distilling mathematical reasoning capabilities into Small Language Models
- **Score**: 27.34
- **Theme**: Small-model reasoning
- **Authors**: Xunyu Zhu et al.
- **Year / venue**: 2024 / Neural Networks
- **URL**: https://openalex.org/W4401263076
- **Topics**: small language models, small models
- **Factual summary seed**: No abstract available.
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 46. Proceedings of the 28th International Conference on Computational Linguistics
- **Score**: 27.3
- **Theme**: Recursive / iterative reasoning
- **Authors**: Xiaodan Zhu et al.
- **Year / venue**: 2020 / n/a
- **URL**: https://openalex.org/W4365799947
- **Topics**: recursive reasoning
- **Factual summary seed**: Only eighteen months ago, I joined Leo, Horacio, Mónica, and Nuria on a tour of the delights that the Barcelona venue would be offering us in September 2020.Together with Chengqing, we made great plans for a fantastic intellectual and social gathering with our colleagues from fa…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 47. Iterative Scene Graph Generation
- **Score**: 27.14
- **Theme**: Recursive / iterative reasoning
- **Authors**: Siddhesh Khandelwal, Leonid Sigal
- **Year / venue**: 2022 / arXiv (Cornell University)
- **URL**: https://openalex.org/W4288724445
- **Topics**: recursive reasoning
- **Factual summary seed**: The task of scene graph generation entails identifying object entities and their corresponding interaction predicates in a given image (or video). Due to the combinatorially large solution space, existing approaches to scene graph generation assume certain factorization of the j…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 48. Reasoning with Latent Structure Refinement for Document-Level Relation Extraction
- **Score**: 26.73
- **Theme**: Recursive / iterative reasoning
- **Authors**: Guoshun Nan et al.
- **Year / venue**: 2020 / n/a
- **URL**: https://openalex.org/W3035053871
- **Topics**: recursive reasoning
- **Factual summary seed**: Document-level relation extraction requires integrating information within and across multiple sentences of a document and capturing complex interactions between inter-sentence entities. However, effective aggregation of relevant information in the document remains a challenging…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 49. Transformers in Vision: A Survey
- **Score**: 26.7
- **Theme**: Recursive / iterative reasoning
- **Authors**: Salman Khan et al.
- **Year / venue**: 2022 / ACM Computing Surveys
- **URL**: https://openalex.org/W3119997354
- **Topics**: recursive reasoning, memory/state-space/sparse
- **Factual summary seed**: Astounding results from Transformer models on natural language tasks have intrigued the vision community to study their application to computer vision problems. Among their salient benefits, Transformers enable modeling long dependencies between input sequence elements and suppo…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 50. Group-Wise Semantic Mining for Weakly Supervised Semantic Segmentation
- **Score**: 26.44
- **Theme**: Recursive / iterative reasoning
- **Authors**: Xueyi Li et al.
- **Year / venue**: 2021 / Proceedings of the AAAI Conference on Artificial Intelligence
- **URL**: https://openalex.org/W3173957243
- **Topics**: small models, curriculum/synthetic/RL, recursive reasoning
- **Factual summary seed**: Acquiring sufficient ground-truth supervision to train deep vi- sual models has been a bottleneck over the years due to the data-hungry nature of deep learning. This is exacerbated in some structured prediction tasks, such as semantic segmen- tation, which requires pixel-level a…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 51. Learning richness modulates equality reasoning in neural networks
- **Score**: 26.3
- **Theme**: Recursive / iterative reasoning
- **Authors**: William L. Tong, Cengiz Pehlevan
- **Year / venue**: 2025 / arXiv
- **URL**: http://arxiv.org/abs/2503.09781v3
- **Topics**: small models, ARC/abstract reasoning, curriculum/synthetic/RL, mechanistic interpretability, recursive reasoning
- **Factual summary seed**: Equality reasoning is ubiquitous and purely abstract: sameness or difference may be evaluated no matter the nature of the underlying objects. As a result, same-different (SD) tasks have been extensively studied as a starting point for understanding abstract reasoning in humans a…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 52. Distributed Cognition for AI-supported Remote Operations: Challenges and Research Directions
- **Score**: 26.3
- **Theme**: Recursive / iterative reasoning
- **Authors**: Rune Møberg Jacobsen et al.
- **Year / venue**: 2025 / arXiv
- **URL**: http://arxiv.org/abs/2504.14996v1
- **Topics**: memory/state-space/sparse, curriculum/synthetic/RL, recursive reasoning
- **Factual summary seed**: This paper investigates the impact of artificial intelligence integration on remote operations, emphasising its influence on both distributed and team cognition. As remote operations increasingly rely on digital interfaces, sensors, and networked communication, AI-driven systems…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 53. Aligning Large and Small Language Models via Chain-of-Thought Reasoning
- **Score**: 26.18
- **Theme**: Small-model reasoning
- **Authors**: Leonardo Ranaldi, André Victor Lucci Freitas, André Freitas
- **Year / venue**: 2024 / Conference of the European Chapter of the Association for Computational Linguistics
- **URL**: https://openalex.org/W4411630221
- **Topics**: latent/CoT reasoning, small language models, small models
- **Factual summary seed**: Chain-of-Thought (CoT) prompting empowers the reasoning abilities of Large Language Models (LLMs), eliciting them to solve complex reasoning tasks in a step-wise manner.However, these abilities appear only in models with billions of parameters, which represent an entry barrier f…
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 54. A foundational model for joint sequence-function multi-species modeling at scale for long-range genomic prediction
- **Score**: 26.16
- **Theme**: Diffusion / denoising LM
- **Authors**: Sam Boshar et al.
- **Year / venue**: 2025 / bioRxiv (Cold Spring Harbor Laboratory)
- **URL**: https://openalex.org/W7117243582
- **Topics**: small models, diffusion LM, diffusion text
- **Factual summary seed**: Genomic prediction and design require models that integrate local sequence features with long-range regulatory dependencies spanning hundreds of kilobases to megabases. Existing approaches have made substantial progress along complementary axes: supervised sequence-to-function m…
- **Implementation relevance**: Relevant to RDLM masking schedules, denoising objectives, and decoding trajectories.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 55. TrTr-CMR: Cross-Modal Reasoning Dual Transformer for Remote Sensing Image Captioning
- **Score**: 26.14
- **Theme**: Recursive / iterative reasoning
- **Authors**: Yinan Wu et al.
- **Year / venue**: 2024 / IEEE Transactions on Geoscience and Remote Sensing
- **URL**: https://openalex.org/W4403183405
- **Topics**: recursive reasoning, memory/state-space/sparse
- **Factual summary seed**: Remote sensing image captioning (RSIC) is an interesting but challenging cross-modal reasoning task for computer vision and natural language processing. Most of the recent popular approaches for RSIC utilize encoder-decoder architectures, which focus on visual features captured…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 56. Enhancing Small Language Models via ChatGPT and Dataset Augmentation
- **Score**: 26.1
- **Theme**: Small-model reasoning
- **Authors**: Tom Pieper et al.
- **Year / venue**: 2024 / Lecture notes in computer science
- **URL**: https://openalex.org/W4402646312
- **Topics**: small models, small language models
- **Factual summary seed**: No abstract available.
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 57. Enhancing the Mathematical Reasoning Ability of Small Language Models through Thought Chain Distillation
- **Score**: 26.1
- **Theme**: Small-model reasoning
- **Authors**: Xiangyu Shu
- **Year / venue**: 2026 / Mathematical Modeling and Algorithm Application
- **URL**: https://www.semanticscholar.org/paper/9a0f314e55a74b854804fd361e86ea5959fc0119
- **Topics**: small models, latent/CoT reasoning, small language models
- **Factual summary seed**: Large language models (LLMs) have demonstrated strong capabilities in reasoning tasks through the Chain of Thought (CoT) prompting technology, but their large scale makes it difficult to deploy them in resource-constrained environments. This paper explores the transfer of the re…
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 58. D-COT: Disciplined Chain-of-Thought Learning for Efficient Reasoning in Small Language Models
- **Score**: 26.1
- **Theme**: Small-model reasoning
- **Authors**: Shunsuke Ubukata
- **Year / venue**: 2026 / arXiv.org
- **URL**: https://www.semanticscholar.org/paper/57509859df4cbd21fbc93d95f94787ec96db2913
- **Topics**: small models, latent/CoT reasoning, small language models
- **Factual summary seed**: Chain-of-Thought (CoT) distillation from Large Language Models (LLMs) often induces"overthinking"in Small Language Models (SLMs), leading to performance degradation and excessive token consumption. In this study, we propose Disciplined Chain-of-Thought (D-CoT), a novel framework…
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 59. Small language models learn enhanced reasoning skills from medical textbooks
- **Score**: 25.89
- **Theme**: Small-model reasoning
- **Authors**: Hyunjae Kim et al.
- **Year / venue**: 2025 / npj Digital Medicine
- **URL**: https://openalex.org/W4410028173
- **Topics**: neuro-symbolic/programs, small language models, small models
- **Factual summary seed**: Small language models (SLM) offer promise for medical applications by addressing the privacy and hardware constraints of large language models; however, their limited parameters (often fewer than ten billion) hinder multi-step reasoning for complex medical tasks. This study pres…
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 60. CaDCR: An Efficient Cascaded Dynamic Collaborative Reasoning Framework for Intelligent Recognition Systems
- **Score**: 25.7
- **Theme**: Recursive / iterative reasoning
- **Authors**: Bowen Li et al.
- **Year / venue**: 2025 / Electronics
- **URL**: https://openalex.org/W4411804431
- **Topics**: small models, neuro-symbolic/programs, curriculum/synthetic/RL, recursive reasoning
- **Factual summary seed**: To address the challenges of high computational cost and energy consumption posed by deep neural networks in embedded systems, this paper presents CaDCR, a lightweight dynamic collaborative reasoning framework. By integrating a feature discrepancy-guided skipping mechanism with…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 61. ThinkSLM: Towards Reasoning in Small Language Models
- **Score**: 25.61
- **Theme**: Small-model reasoning
- **Authors**: Gaurav Srivastava, Shuxiang Cao, Xuan Wang
- **Year / venue**: 2025 / n/a
- **URL**: https://openalex.org/W4416035376
- **Topics**: small models, small language models
- **Factual summary seed**: Reasoning has long been viewed as an emergent property of large language models (LLMs).However, recent studies challenge this assumption, showing that small language models (SLMs) can also achieve competitive reasoning performance.This paper introduces THINKSLM, the first extens…
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 62. Simple and Effective Masked Diffusion Language Models
- **Score**: 25.54
- **Theme**: Diffusion / denoising LM
- **Authors**: Marianne Arriola et al.
- **Year / venue**: 2024 / n/a
- **URL**: https://openalex.org/W4415800813
- **Topics**: diffusion LM, diffusion text
- **Factual summary seed**: No abstract available.
- **Implementation relevance**: Relevant to RDLM masking schedules, denoising objectives, and decoding trajectories.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 63. Teaching Small Language Models Reasoning through Counterfactual Distillation
- **Score**: 24.9
- **Theme**: Small-model reasoning
- **Authors**: Tao Feng et al.
- **Year / venue**: 2024 / n/a
- **URL**: https://openalex.org/W4404784411
- **Topics**: small models, small language models
- **Factual summary seed**: No abstract available.
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 64. Cognitive Dissonance Artificial Intelligence (CD-AI): The Mind at War with Itself. Harnessing Discomfort to Sharpen Critical Thinking
- **Score**: 24.8
- **Theme**: Recursive / iterative reasoning
- **Authors**: Delia Deliu
- **Year / venue**: 2025 / arXiv
- **URL**: http://arxiv.org/abs/2507.08804v1
- **Topics**: curriculum/synthetic/RL, recursive reasoning
- **Factual summary seed**: AI-augmented systems are traditionally designed to streamline human decision-making by minimizing cognitive load, clarifying arguments, and optimizing efficiency. However, in a world where algorithmic certainty risks becoming an Orwellian tool of epistemic control, true intellec…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 65. Integrating semantic retrieval and chain-of-thought reasoning in small language models for SNOMED CT normalization
- **Score**: 24.6
- **Theme**: Small-model reasoning
- **Authors**: P. López-Úbeda, T. Martín-Noguerol, A. Luna
- **Year / venue**: 2026 / Int. J. Medical Informatics
- **URL**: https://www.semanticscholar.org/paper/83199c0c53304c6f68006d918b20eeefad863e5e
- **Topics**: small models, latent/CoT reasoning, neuro-symbolic/programs, small language models
- **Factual summary seed**: BACKGROUND AND OBJECTIVE Breast lesion biopsy assessment generates a high volume of pathology reports, posing a significant workload for pathologists. Standardized coding systems such as SNOMED CT Morphological codes enable consistent documentation, facilitate accurate data shar…
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 66. Asynchronous Blockchain Recording for Chain-of-Thought Tracing in Small Language Models
- **Score**: 24.6
- **Theme**: Small-model reasoning
- **Authors**: Sungmoon Park et al.
- **Year / venue**: 2026 / Digital Signal Processing and Signal Processing Education Workshop
- **URL**: https://www.semanticscholar.org/paper/a3bfe6163b73686365a07df63082268e2cd1c386
- **Topics**: small models, latent/CoT reasoning, curriculum/synthetic/RL, small language models
- **Factual summary seed**: Small Language Models (sLMs) require transparency in their reasoning processes for deployment in critical domains. This paper presents an asynchronous blockchain recording system for tracking Chain-of-Thought (CoT) reasoning in sLMs. We implement a system using the Qwen2.5-1.5B…
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 67. Multimodal Transformer With Multi-View Visual Representation for Image Captioning
- **Score**: 24.47
- **Theme**: Recursive / iterative reasoning
- **Authors**: Jun Yu et al.
- **Year / venue**: 2019 / IEEE Transactions on Circuits and Systems for Video Technology
- **URL**: https://openalex.org/W2981165461
- **Topics**: recursive reasoning
- **Factual summary seed**: Image captioning aims to automatically generate a natural language description of a given image, and most state-of-the-art models have adopted an encoder-decoder framework. The framework consists of a convolution neural network (CNN)-based image encoder that extracts region-base…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 68. Geometric Point Attention Transformer for 3D Shape Reassembly
- **Score**: 24.31
- **Theme**: Recursive / iterative reasoning
- **Authors**: Jiahan Li et al.
- **Year / venue**: 2024 / arXiv.org
- **URL**: https://www.semanticscholar.org/paper/55c4ea2143939c7162990e5491b6dffcf63308ca
- **Topics**: recursive reasoning
- **Factual summary seed**: Shape assembly, which aims to reassemble separate parts into a complete object, has gained significant interest in recent years. Existing methods primarily rely on networks to predict the poses of individual parts, but often fail to effectively capture the geometric interactions…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 69. Predicting concentration levels of air pollutants by transfer learning and recurrent neural network
- **Score**: 24.3
- **Theme**: Recursive / iterative reasoning
- **Authors**: Iat Hang Fong et al.
- **Year / venue**: 2025 / arXiv
- **URL**: http://arxiv.org/abs/2502.01654v1
- **Topics**: recursive reasoning, neuro-symbolic/programs, memory/state-space/sparse
- **Factual summary seed**: Air pollution (AP) poses a great threat to human health, and people are paying more attention than ever to its prediction. Accurate prediction of AP helps people to plan for their outdoor activities and aids protecting human health. In this paper, long-short term memory (LSTM) r…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 70. Hierarchical Reasoning Models: Perspectives and Misconceptions
- **Score**: 24.3
- **Theme**: Recursive / iterative reasoning
- **Authors**: Renee Ge, Qianli Liao, Tomaso Poggio
- **Year / venue**: 2025 / arXiv
- **URL**: http://arxiv.org/abs/2510.00355v2
- **Topics**: recursive reasoning, neuro-symbolic/programs, curriculum/synthetic/RL
- **Factual summary seed**: Transformers have demonstrated remarkable performance in natural language processing and related domains, as they largely focus on sequential, autoregressive next-token prediction tasks. Yet, they struggle in logical reasoning, not necessarily because of a fundamental limitation…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 71. Steering Masked Discrete Diffusion Models via Discrete Denoising Posterior Prediction
- **Score**: 23.93
- **Theme**: Diffusion / denoising LM
- **Authors**: Jarrid Rector-Brooks et al.
- **Year / venue**: 2024 / arXiv (Cornell University)
- **URL**: https://openalex.org/W4403365366
- **Topics**: diffusion LM, curriculum/synthetic/RL, diffusion text
- **Factual summary seed**: Generative modeling of discrete data underlies important applications spanning text-based agents like ChatGPT to the design of the very building blocks of life in protein sequences. However, application domains need to exert control over the generated data by steering the genera…
- **Implementation relevance**: Relevant to RDLM masking schedules, denoising objectives, and decoding trajectories.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 72. Less is More: Recursive Reasoning with Tiny Networks
- **Score**: 23.9
- **Theme**: Recursive / iterative reasoning
- **Authors**: Alexia Jolicoeur‐Martineau
- **Year / venue**: 2025 / Research Square
- **URL**: https://openalex.org/W4417455104
- **Topics**: recursive reasoning, small models
- **Factual summary seed**: No abstract available.
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 73. Intelligent Interaction Strategies for Context-Aware Cognitive Augmentation
- **Score**: 23.8
- **Theme**: Recursive / iterative reasoning
- **Authors**:  Xiangrong et al.
- **Year / venue**: 2025 / arXiv
- **URL**: http://arxiv.org/abs/2504.13684v1
- **Topics**: curriculum/synthetic/RL, recursive reasoning
- **Factual summary seed**: Human cognition is constrained by processing limitations, leading to cognitive overload and inefficiencies in knowledge synthesis and decision-making. Large Language Models (LLMs) present an opportunity for cognitive augmentation, but their current reactive nature limits their r…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 74. Chain of Thought Strategy for Smaller LLMs for Medical Reasoning
- **Score**: 23.73
- **Theme**: Small-model reasoning
- **Authors**: Hurmat Ali Shah, Mowafa Househ
- **Year / venue**: 2025 / Studies in health technology and informatics
- **URL**: https://openalex.org/W4410447959
- **Topics**: latent/CoT reasoning, mechanistic interpretability, small language models
- **Factual summary seed**: This paper investigates the application of Chain of Thought (CoT) reasoning to enhance the performance of smaller language models in medical question-answering tasks. By leveraging CoT prompting strategies, we aim to improve model accuracy and interpretability, especially in res…
- **Implementation relevance**: Relevant to keeping model size/training cost low while improving reasoning quality.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

### 75. Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications
- **Score**: 23.6
- **Theme**: Recursive / iterative reasoning
- **Authors**: Wojciech Samek et al.
- **Year / venue**: 2021 / Proceedings of the IEEE
- **URL**: https://openalex.org/W3132191748
- **Topics**: mechanistic interpretability, recursive reasoning
- **Factual summary seed**: With the broader and highly successful usage of machine learning (ML) in industry and the sciences, there has been a growing demand for explainable artificial intelligence (XAI). Interpretability and explanation methods for gaining a better understanding of the problem-solving a…
- **Implementation relevance**: Relevant to TRM-style recurrent refinement and adaptive computation depth.
- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.

## Categorized bibliography

### ARC / program / symbolic reasoning

- **ATHENA: A Cognitive Architecture for Persistent Execution and Skill Formation in Autonomous Agents** (2026, score=50.1, cites=0) — Hafizur Rahman. https://openalex.org/W7124259583
- **Hierarchical Reasoning Model** (2025, score=47.8) — Guan Wang et al.. http://arxiv.org/abs/2506.21734v3
- **Dynamical Systems Theory Behind a Hierarchical Reasoning Model** (2026, score=33.1, cites=0) — Vasiliy A. Es'kin, Mikhail E. Smorkalov. http://arxiv.org/abs/2603.22871v1

### Curriculum / synthetic data / RL

- **Deep multiagent reinforcement learning: challenges and directions** (2022, score=27.56, cites=165) — Annie Wong et al.. https://openalex.org/W4306786778

### Diffusion / denoising LM

- **Simplified and Generalized Masked Diffusion for Discrete Data** (2024, score=30.81, cites=3) — Jiaxin Shi et al.. https://openalex.org/W4399455367
- **Diffutron: A Masked Diffusion Language Model for Turkish Language** (2026, score=29.6, cites=0) — Şuayp Talha Kocabay, Talha Rüzgar Akkuş. https://openalex.org/W7140346153
- **A foundational model for joint sequence-function multi-species modeling at scale for long-range genomic prediction** (2025, score=26.16, cites=8) — Sam Boshar et al.. https://openalex.org/W7117243582
- **Simple and Effective Masked Diffusion Language Models** (2024, score=25.54, cites=6) — Marianne Arriola et al.. https://openalex.org/W4415800813
- **Steering Masked Discrete Diffusion Models via Discrete Denoising Posterior Prediction** (2024, score=23.93, cites=2) — Jarrid Rector-Brooks et al.. https://openalex.org/W4403365366
- **Structure Language Models for Protein Conformation Generation** (2024, score=23.1, cites=4) — Jiarui Lu et al.. https://openalex.org/W4404306486
- **CodeDiffuSe: A masked diffusion framework for structure-aware code completion and repair** (2025, score=20.23, cites=2) — Aytuğ Onan, Hesham Alhumyani. https://openalex.org/W4414466491

### Recursive / iterative reasoning

- **One Step Forward and K Steps Back: Better Reasoning with Denoising Recursion Models** (2026, score=51.1, cites=0) — Chris Cameron et al.. https://www.semanticscholar.org/paper/b433acdd80aed14800400721a2e9edcaf1dc8928
- **Symbol-Equivariant Recurrent Reasoning Models** (2026, score=50.1, cites=0) — Richard Freinschlag et al.. https://openalex.org/W7133364588
- **Closed-Loop Transformers: Autoregressive Modeling as Iterative Latent Equilibrium** (2025, score=48.73, cites=2) — Akbar Anbar Jafari, G. Anbarjafari. https://www.semanticscholar.org/paper/cfefe7a5df5b04883ef63bfb7dd675682d6b0085
- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (2023, score=41.2, cites=984) — Albert Gu, Tri Dao. https://openalex.org/W4389326242
- **Multimodal Latent Reasoning via Hierarchical Visual Cues Injection** (2026, score=36.6, cites=0) — Yiming Zhang et al.. https://www.semanticscholar.org/paper/30e7b2338f9fada55addb68e870803975da2bcc2
- **Virtual Parameter Sharpening: Dynamic Low-Rank Perturbations for Inference-Time Reasoning Enhancement** (2025, score=34.3, cites=0) — Saba Kublashvili. https://www.semanticscholar.org/paper/d1dafc2dba42c7702f31083b4fe3fcc32d762537
- **Artificial Intelligence for Materials Discovery, Development, and Optimization** (2025, score=33.12, cites=86) — Benediktus Madika et al.. https://openalex.org/W4412654790
- **Toward expert-level medical question answering with large language models** (2025, score=31.8, cites=647) — K. K. Singhal et al.. https://openalex.org/W4406152279
- **PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation** (2024, score=30.81, cites=58) — Qiyao Xue et al.. https://www.semanticscholar.org/paper/015b1f127b6c31654e3597b75876eed8e445d866
- **A Comprehensive Evaluation of Transformer-Based Question Answering Models and RAG-Enhanced Design** (2025, score=30.8, cites=0) — Zichen Zhang et al.. https://www.semanticscholar.org/paper/8142b1dbd9163f9e5166da2d25c1c9c1f90c74cb
- **Revisiting Large Language Models as Zero-shot Relation Extractors** (2023, score=30.57, cites=41) — Guozheng Li, Peng Wang, Wenjun Ke. https://openalex.org/W4389518985
- **Image Super-Resolution Via Iterative Refinement** (2022, score=30.4, cites=1621) — Chitwan Saharia et al.. https://openalex.org/W3155072588
- **Towards Expert-Level Medical Question Answering with Large Language Models** (2023, score=29.77, cites=333) — Karan Singhal et al.. https://openalex.org/W4377009978
- **A survey on large language model based autonomous agents** (2024, score=29.5, cites=1037) — Lei Wang et al.. https://openalex.org/W4393065402
- **Object Detection in 20 Years: A Survey** (2019, score=29.5, cites=560) — Zhengxia Zou et al.. https://openalex.org/W2944165510
- **Embodied Reasoning with Self-Feedback** (2024, score=29.43, cites=2) — Pranav Kak, Sushma Jain. https://www.semanticscholar.org/paper/4c2426f47d90839301899a28117c71fae3179b68
- **Graph of Thoughts: Solving Elaborate Problems with Large Language Models** (2024, score=29.27, cites=387) — Maciej Besta et al.. https://openalex.org/W4393160302
- **Multi-Hop Knowledge Graph Reasoning with Reward Shaping** (2018, score=29.23, cites=375) — Xi Lin, Richard Socher, Caiming Xiong. https://openalex.org/W2889344053
- **MizAR 60 for Mizar 50** (2023, score=29.2, cites=75659) — Jakubův, Jan et al.. https://openalex.org/W4385245566
- **One-Shot Autoregressive Generation of Combinatorial Optimization Solutions Based on the Large Language Model Architecture and Learning Algorithms** (2025, score=29.2, cites=1) — Bishad Ghimire, Ausif Mahmood, Khaled Elleithy. https://openalex.org/W4408901775
- **Weight-Tied Adaptive Recursive Vision–Language–Action Transformer for Efficient Multimodal Robotic Control** (2026, score=28.6, cites=0) — Howaida Allam, Inam Ullah Khan. https://www.semanticscholar.org/paper/d12bf0d70d5c748bf5354542efc098db26b24b9f
- **A comprehensive survey on pretrained foundation models: a history from BERT to ChatGPT** (2024, score=28.17, cites=359) — Ce Zhou et al.. https://openalex.org/W4404658388
- **Applications of Artificial Intelligence in Transport: An Overview** (2019, score=28.0, cites=720) — Rusul Abduljabbar et al.. https://openalex.org/W2908162093
- **Uncovering Graph Reasoning in Decoder-only Transformers with Circuit Tracing** (2025, score=27.8) — Xinnan Dai et al.. http://arxiv.org/abs/2509.20336v1
- **Recurrent Memory Transformer** (2022, score=27.69, cites=26) — Aydar Bulatov, Yuri Kuratov, Mikhail Burtsev. https://openalex.org/W4285595435
- **Mastering Long-Context Multi-Task Reasoning with Transformers and Recurrent Memory** (2024, score=27.4, cites=1) — Aleksandr Bulatov, Yuri Kuratov, Mikhail Burtsev. https://openalex.org/W4406757829
- **Proceedings of the 28th International Conference on Computational Linguistics** (2020, score=27.3, cites=1422) — Xiaodan Zhu et al.. https://openalex.org/W4365799947
- **Iterative Scene Graph Generation** (2022, score=27.14, cites=11) — Siddhesh Khandelwal, Leonid Sigal. https://openalex.org/W4288724445
- **Reasoning with Latent Structure Refinement for Document-Level Relation Extraction** (2020, score=26.73, cites=298) — Guoshun Nan et al.. https://openalex.org/W3035053871
- **Transformers in Vision: A Survey** (2022, score=26.7, cites=125) — Salman Khan et al.. https://openalex.org/W3119997354
- **Group-Wise Semantic Mining for Weakly Supervised Semantic Segmentation** (2021, score=26.44, cites=129) — Xueyi Li et al.. https://openalex.org/W3173957243
- **Learning richness modulates equality reasoning in neural networks** (2025, score=26.3) — William L. Tong, Cengiz Pehlevan. http://arxiv.org/abs/2503.09781v3
- **Distributed Cognition for AI-supported Remote Operations: Challenges and Research Directions** (2025, score=26.3) — Rune Møberg Jacobsen et al.. http://arxiv.org/abs/2504.14996v1
- **TrTr-CMR: Cross-Modal Reasoning Dual Transformer for Remote Sensing Image Captioning** (2024, score=26.14, cites=23) — Yinan Wu et al.. https://openalex.org/W4403183405
- **CaDCR: An Efficient Cascaded Dynamic Collaborative Reasoning Framework for Intelligent Recognition Systems** (2025, score=25.7, cites=1) — Bowen Li et al.. https://openalex.org/W4411804431
- **Cognitive Dissonance Artificial Intelligence (CD-AI): The Mind at War with Itself. Harnessing Discomfort to Sharpen Critical Thinking** (2025, score=24.8) — Delia Deliu. http://arxiv.org/abs/2507.08804v1
- **Multimodal Transformer With Multi-View Visual Representation for Image Captioning** (2019, score=24.47, cites=454) — Jun Yu et al.. https://openalex.org/W2981165461
- **Geometric Point Attention Transformer for 3D Shape Reassembly** (2024, score=24.31, cites=3) — Jiahan Li et al.. https://www.semanticscholar.org/paper/55c4ea2143939c7162990e5491b6dffcf63308ca
- **Predicting concentration levels of air pollutants by transfer learning and recurrent neural network** (2025, score=24.3) — Iat Hang Fong et al.. http://arxiv.org/abs/2502.01654v1
- **Hierarchical Reasoning Models: Perspectives and Misconceptions** (2025, score=24.3) — Renee Ge, Qianli Liao, Tomaso Poggio. http://arxiv.org/abs/2510.00355v2
- **Less is More: Recursive Reasoning with Tiny Networks** (2025, score=23.9, cites=4) — Alexia Jolicoeur‐Martineau. https://openalex.org/W4417455104
- **Intelligent Interaction Strategies for Context-Aware Cognitive Augmentation** (2025, score=23.8) —  Xiangrong et al.. http://arxiv.org/abs/2504.13684v1
- **Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications** (2021, score=23.6, cites=1281) — Wojciech Samek et al.. https://openalex.org/W3132191748
- **Shots and Boosters: Exploring the Use of Combined Prebunking Interventions to Raise Critical Thinking and Create Long-Term Protection Against Misinformation** (2025, score=23.3) — Huiyun Tang, Anastasia Sergeeva. http://arxiv.org/abs/2505.07486v1
- **Navigating the State of Cognitive Flow: Context-Aware AI Interventions for Effective Reasoning Support** (2025, score=23.3) — Dinithi Dissanayake, Suranga Nanayakkara. http://arxiv.org/abs/2504.16021v1
- **Enhancing Critical Thinking with AI: A Tailored Warning System for RAG Models** (2025, score=22.3) — Xuyang Zhu, Sejoon Chang, Andrew Kuik. http://arxiv.org/abs/2504.16883v1
- **A Review of Artificial Intelligence (AI) in Education from 2010 to 2020** (2021, score=22.1, cites=1100) — Xuesong Zhai et al.. https://openalex.org/W3156614709
- **Multivariate Adaptive Regression Splines** (1991, score=22.0, cites=8060) — Jerome H. Friedman. https://openalex.org/W2102201073
- **Demonstrative and non-demonstrative reasoning by analogy** (2008, score=21.5) — Emiliano Ippoliti. http://arxiv.org/abs/0810.5078v1
- **Value-Decomposition Multi-Agent Actor-Critics** (2021, score=21.46, cites=89) — Jianyu Su, Stephen Adams, Peter A. Beling. https://openalex.org/W3176265013
- **Recurrent Vision Transformer for Solving Visual Reasoning Problems** (2022, score=21.44, cites=6) — Nicola Messina et al.. https://openalex.org/W3215520958
- **Inferring on the Intentions of Others by Hierarchical Bayesian Learning** (2014, score=21.09, cites=229) — Andreea O. Diaconescu et al.. https://openalex.org/W2076243793
- **Dual Accuracy-Quality-Driven Neural Network for Prediction Interval Generation** (2022, score=20.9) — Giorgio Morales, John W. Sheppard. http://arxiv.org/abs/2212.06370v4
- **Recursive Non-Autoregressive Graph-to-Graph Transformer for Dependency Parsing with Iterative Refinement** (2020, score=20.8) — Alireza Mohammadshahi, James Henderson. http://arxiv.org/abs/2003.13118v2
- **Natural Language Processing (almost) from Scratch** (2011, score=20.5, cites=3991) — Ronan Collobert et al.. https://openalex.org/W2952230511
- **A semantic matching energy function for learning with multi-relational data** (2013, score=20.5, cites=683) — Antoine Bordes et al.. https://openalex.org/W68132019
- **QAngaroo (MedHop + WikiHop) - Constructing Datasets for Multi-hop Reading Comprehension Across Documents** (2018, score=20.5, cites=489) — Johannes Welbl, Pontus Stenetorp, Sebastian Riedel. https://openalex.org/W2963866616
- **Autonomous agents modelling other agents: A comprehensive survey and open problems** (2018, score=20.36, cites=415) — Stefano V. Albrecht, Peter Stone. https://openalex.org/W2758442112
- **MMTC-Hinton-Sutton** (2025, score=20.3, cites=0) — Nicolaii Blavatsky. https://openalex.org/W7111960574
- **Towards Explainable Neural-Symbolic Visual Reasoning** (2019, score=20.0) — Adrien Bennetot et al.. http://arxiv.org/abs/1909.09065v2
- **From Propositional Logic to Plausible Reasoning: A Uniqueness Theorem** (2017, score=20.0) — Kevin S. Van Horn. http://arxiv.org/abs/1706.05261v1
- **PonderNet: Learning to Ponder** (2021, score=19.63, cites=21) — Andrea Banino, Jan Balaguer, Charles Blundell. https://openalex.org/W3182814358
- **Multi-source knowledge fusion: a survey** (2020, score=19.25, cites=95) — Xiaojuan Zhao et al.. https://openalex.org/W3016026271
- **Theory of mind and decision science: Towards a typology of tasks and computational models** (2020, score=18.91, cites=73) — Tessa Rusch et al.. https://openalex.org/W3025570388
- **Midterm Status Report of the ILC Technology Network Activities** (2026, score=18.1) — ILC Technology Network. http://arxiv.org/abs/2603.01172v1
- **The Deep Arbitrary Polynomial Chaos Neural Network or how Deep Artificial Neural Networks could benefit from Data-Driven Homogeneous Chaos Theory** (2023, score=15.7) — Sergey Oladyshkin et al.. http://arxiv.org/abs/2306.14753v1
- **Parallel Neural Networks in Golang** (2023, score=15.7) — Daniela Kalwarowskyj, Erich Schikuta. http://arxiv.org/abs/2304.09590v1
- **A Review on Neural Network Models of Schizophrenia and Autism Spectrum Disorder** (2019, score=14.0) — Pablo Lanillos et al.. http://arxiv.org/abs/1906.10015v2
- **Masked Conditional Neural Networks for Audio Classification** (2018, score=14.0) — Fady Medhat, David Chesmore, John Robinson. http://arxiv.org/abs/1803.02421v2
- **A Tutorial about Random Neural Networks in Supervised Learning** (2016, score=14.0) — Sebastián Basterrech, Gerardo Rubino. http://arxiv.org/abs/1609.04846v1
- **Development of a sensory-neural network for medical diagnosing** (2018, score=12.5) — Igor Grabec, Eva Švegl, Mihael Sok. http://arxiv.org/abs/1807.02477v1
- **Probabilistic Reasoning via Deep Learning: Neural Association Models** (2016, score=12.5) — Quan Liu et al.. http://arxiv.org/abs/1603.07704v2
- **How transferable are features in deep neural networks?** (2014, score=12.5) — Jason Yosinski et al.. http://arxiv.org/abs/1411.1792v1

### Small-model reasoning

- **rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking** (2025, score=38.16, cites=8) — Xinyu Guan et al.. https://openalex.org/W4406231568
- **A Survey of Large Language Models** (2026, score=32.6, cites=1392) — Wayne Xin Zhao et al.. https://openalex.org/W4362515116
- **Beyond Test-Time Compute Strategies: Advocating Energy-per-Token in LLM Inference** (2025, score=32.4, cites=4) — Pascal Wilhelm, Thorsten Wittkopp, Odej Kao. https://openalex.org/W4409047589
- **Neural-Symbolic Collaborative Distillation: Advancing Small Language Models for Complex Reasoning Tasks** (2025, score=32.11, cites=3) — Huanxuan Liao et al.. https://openalex.org/W4409348057
- **Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing** (2022, score=30.4, cites=3536) — Pengfei Liu et al.. https://openalex.org/W3185341429
- **MAGDi: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models** (2024, score=30.4, cites=1) — Justin Chih-Yao Chen et al.. https://openalex.org/W4391556796
- **Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning** (2024, score=30.31, cites=3) — Amrith Setlur et al.. https://openalex.org/W4403365357
- **Can Small Language Models Help Large Language Models Reason Better?: LM-Guided Chain-of-Thought** (2024, score=29.9, cites=19) — Jooyoung Lee et al.. https://www.semanticscholar.org/paper/f19ddba513dfeca571a6e2ba7542a63677cd5e3b
- **Knowledge-Augmented Reasoning Distillation for Small Language Models in Knowledge-Intensive Tasks** (2023, score=29.23, cites=21) — Minki Kang et al.. https://openalex.org/W4378942311
- **Distilling Reasoning Capabilities into Smaller Language Models** (2023, score=29.1, cites=62) — Kumar Shridhar, Alessandro Stolfo, Mrinmaya Sachan. https://openalex.org/W4385571831
- **A Metaverse: Taxonomy, Components, Applications, and Open Challenges** (2022, score=28.9, cites=1729) — Sangmin Park, Young‐Gab Kim. https://openalex.org/W4206484811
- **Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning** (2024, score=27.88, cites=288) — Hao Shao et al.. https://www.semanticscholar.org/paper/ee8c1a46c90f1261c23479e15c6bed7f67ad8943
- **Distilling mathematical reasoning capabilities into Small Language Models** (2024, score=27.34, cites=12) — Xunyu Zhu et al.. https://openalex.org/W4401263076
- **Aligning Large and Small Language Models via Chain-of-Thought Reasoning** (2024, score=26.18, cites=77) — Leonardo Ranaldi, André Victor Lucci Freitas, André Freitas. https://openalex.org/W4411630221
- **Enhancing Small Language Models via ChatGPT and Dataset Augmentation** (2024, score=26.1, cites=4) — Tom Pieper et al.. https://openalex.org/W4402646312
- **Enhancing the Mathematical Reasoning Ability of Small Language Models through Thought Chain Distillation** (2026, score=26.1, cites=0) — Xiangyu Shu. https://www.semanticscholar.org/paper/9a0f314e55a74b854804fd361e86ea5959fc0119
- **D-COT: Disciplined Chain-of-Thought Learning for Efficient Reasoning in Small Language Models** (2026, score=26.1, cites=0) — Shunsuke Ubukata. https://www.semanticscholar.org/paper/57509859df4cbd21fbc93d95f94787ec96db2913
- **Small language models learn enhanced reasoning skills from medical textbooks** (2025, score=25.89, cites=33) — Hyunjae Kim et al.. https://openalex.org/W4410028173
- **ThinkSLM: Towards Reasoning in Small Language Models** (2025, score=25.61, cites=3) — Gaurav Srivastava, Shuxiang Cao, Xuan Wang. https://openalex.org/W4416035376
- **Teaching Small Language Models Reasoning through Counterfactual Distillation** (2024, score=24.9, cites=1) — Tao Feng et al.. https://openalex.org/W4404784411
- **Integrating semantic retrieval and chain-of-thought reasoning in small language models for SNOMED CT normalization** (2026, score=24.6, cites=0) — P. López-Úbeda, T. Martín-Noguerol, A. Luna. https://www.semanticscholar.org/paper/83199c0c53304c6f68006d918b20eeefad863e5e
- **Asynchronous Blockchain Recording for Chain-of-Thought Tracing in Small Language Models** (2026, score=24.6, cites=0) — Sungmoon Park et al.. https://www.semanticscholar.org/paper/a3bfe6163b73686365a07df63082268e2cd1c386
- **Chain of Thought Strategy for Smaller LLMs for Medical Reasoning** (2025, score=23.73, cites=2) — Hurmat Ali Shah, Mowafa Househ. https://openalex.org/W4410447959
- **DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning in Language Models** (2023, score=23.31, cites=15) — Zheng Ge et al.. https://openalex.org/W4387994822
- **Optimizing Small Language Models for NL2SQL via Chain-of-Thought Fine-Tuning** (2026, score=23.0, cites=1) — Anshul Solanki et al.. https://www.semanticscholar.org/paper/3cf374383750f3495de85509261b7a24f05124fc
- **Reasoning Aware Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling** (2024, score=22.81, cites=3) — Guoyang Wan et al.. https://openalex.org/W4403556398
- **Towards a small language model powered chain‐of‐reasoning for open‐domain question answering** (2024, score=21.71, cites=7) — Jihyeon Roh, Minho Kim, Kyoungman Bae. https://openalex.org/W4392245172
- **Knowledge Distillation-Enhanced Behavior Transformer for Decision-Making of Autonomous Driving** (2025, score=21.63, cites=5) — Rui Zhao et al.. https://openalex.org/W4405965591
- **Multimodal Learning With Transformers: A Survey** (2023, score=21.2, cites=829) — Peng Xu, Xiatian Zhu, David A. Clifton. https://openalex.org/W4376226279
- **Cognitive Edge Computing: A Comprehensive Survey on Optimizing Large Models and AI Agents for Pervasive Deployment** (2025, score=21.11, cites=3) — Xubin Wang et al.. https://openalex.org/W4406167404
- **Reading Between the Lines: Commonsense Reasoning in Small Language Models** (2024, score=21.1, cites=4) — Wasif Feroze et al.. https://openalex.org/W4409097773
- **From Chain to Loop: Improving Reasoning Capability in Small Language Models via Loop-of-Thought** (2025, score=20.8, cites=0) — Mingxin Ji et al.. https://www.semanticscholar.org/paper/6617b5f1fb9962e499782cafdfd17fbd14259fb3
- **RWKV: Reinventing RNNs for the Transformer Era** (2023, score=20.62, cites=296) — Bo Peng et al.. https://openalex.org/W4389524555
- **Scaling Instruction-Finetuned Language Models** (2022, score=20.4, cites=1187) — Hyung Won Chung et al.. https://openalex.org/W4307079201
- **Reverse Thinking Makes LLMs Stronger Reasoners** (2024, score=19.4, cites=1) — Justin Chih-Yao Chen et al.. https://openalex.org/W4405031736
- **ChatGPT for good? On opportunities and challenges of large language models for education** (2023, score=18.7, cites=4696) — Enkelejda Kasneci et al.. https://openalex.org/W4323655724
- **Overview of the Transformer-based Models for NLP Tasks** (2020, score=18.52, cites=373) — Anthony Gillioz et al.. https://openalex.org/W3092557781
- **Muppet: Massive Multi-task Representations with Pre-Finetuning** (2021, score=18.34, cites=176) — Armen Aghajanyan et al.. https://openalex.org/W3124687886
- **LLM-augmented hierarchical reinforcement learning for human-like decision-making of autonomous driving** (2025, score=18.24, cites=13) — Lin Li et al.. https://openalex.org/W4411866407
- **Better &amp; Faster Large Language Models via Multi-token Prediction** (2024, score=15.86, cites=8) — Fabian Gloeckle et al.. https://openalex.org/W4396821494
- **Specializing Smaller Language Models towards Multi-Step Reasoning** (2023, score=15.13, cites=43) — Yao Fu et al.. https://openalex.org/W4318719086
- **PHYBench: Holistic Evaluation of Physical Perception and Reasoning in Large Language Models** (2025, score=13.2, cites=1) — Shi Qiu et al.. https://openalex.org/W4414634788

