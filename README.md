<h1 align="center">🤖 KERV: Kinematic-Rectified Speculative Decoding for Embodied VLA Models</h1>

<p align="center">
  Zihao Zheng · <a href="https://github.com/lusunn111">Zhihao Mao</a> · Maoliang Li · Jiayu Chen · Xinhao Sun ·<br>
  Zhaobo Zhang · Donggang Cao · Hong Mei · Xiang Chen
</p>

<p align="center">
  <a href="https://dac.com/">
    <img src="https://img.shields.io/badge/DAC-2026-7B2CBF.svg" alt="DAC 2026">
  </a>
  <a href="https://arxiv.org/abs/2603.01581">
    <img src="https://img.shields.io/badge/arXiv-2603.01581-B31B1B.svg?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://arxiv.org/pdf/2603.01581">
    <img src="https://img.shields.io/badge/Paper-PDF-E67E22.svg" alt="Paper PDF">
  </a>
  <a href="https://github.com/lusunn111/KERV">
    <img src="https://img.shields.io/badge/Code-GitHub-181717.svg?logo=github&logoColor=white" alt="Code">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT%20%2F%20Apache--2.0-2E86C1.svg" alt="License">
  </a>
</p>

## News

- **2026-08** — 🚀 We open-sourced KERV and its runtime optimization.
- **2026-07** — 🤝 We partnered with BAAI to optimize KERV based on FlagOS.
- **2026-07** — ⚙️ We added runtime optimizations to KERV.
- **2026-03** — 🔥 KERV has been accepted by **DAC 2026**!

KERV is a speculative decoding framework for accelerating autoregressive
Vision-Language-Action (VLA) models. It connects token-domain VLA generation
with kinematic-domain robot control to address two limitations of speculative
decoding in embodied tasks:

1. rejected draft tokens normally trigger expensive re-inference; and
2. a fixed relaxed acceptance threshold cannot adapt to changing robot states,
   tasks, and environments.

KERV uses a Kalman-filter-based mechanism to compensate for rejected action
tokens without re-inference and adjusts the acceptance threshold using
kinematic variability. The on-device runtime further combines CPU-GPU
collaborative execution, static tree-based verification, and hardware-aware
operator fusion.

![Overview of KERV](assets/kerv_overview.png)

The core KERV method was accepted at DAC 2026. This repository also includes
the extended on-device runtime implementation: CPU-GPU collaborative
deployment, static tree-based verification, CUDA Graph replay, persistent
caches, and hardware-aware operator fusion.

> This repository is a research release. It contains the KERV inference path,
> optimized operators, runtime integration, and drafter training pipeline.
> Model weights, generated training samples, datasets, profiler traces, and
> robot assets are not distributed with the source code.

## Overview

An autoregressive VLA model produces a seven-DoF action slice as a sequence of
action tokens. A lightweight drafter proposes candidate tokens, and the VLA
model verifies them in parallel. KERV augments this process with kinematic
feedback.

The verification and draft models execute on the GPU, while the lightweight
Kalman compensation and threshold adjustment run on the CPU. Only the required
control values are exchanged between the two sides.

## Method

![KF-based compensation and kinematic-based threshold adjustment](assets/kerv_mechanisms.png)

### KF-Based Compensation Mechanism

When speculative verification encounters an incorrect action token, standard
methods discard the remaining draft and invoke the VLA model again. KERV
instead maps the accepted token prefix to continuous robot actions, maintains
DoF-grained action histories, and uses a Kalman filter to predict the remainder
of the current action slice. The accepted VLA actions and predicted actions are
concatenated into one complete command, avoiding costly re-inference.

The default setting uses an action context of 10 and a prediction length of 1.
To prevent error accumulation, compensation is activated intermittently; the
following four inference steps use speculative decoding without Kalman
compensation.

### Kinematic-Based Threshold Adjustment

A single relaxed threshold is not reliable across different stages of a robot
trajectory. KERV measures the difference between accepted draft tokens and
verifier tokens, maps the token discrepancy into kinematic variability, and
uses the variability to update the acceptance threshold online. Task- and
robot-specific bounds are obtained through offline pre-sampling; the paper uses
`r_max = 15` and `r_min = 5` for most tasks.

This mechanism preserves the speed benefit of relaxed verification while
rejecting errors that appear small in token space but are unsafe in kinematic
space.

### On-Device Runtime Optimization

KERV includes three complementary runtime optimizations:

- **CPU-GPU collaborative deployment.** Draft and verification inference run
  on the GPU; Kalman compensation and threshold adjustment run on the CPU.
- **Static tree-based verification.** Offline acceptance statistics are used to
  preset a depth-5 verification tree. Node layouts, parent-child relations,
  attention masks, position IDs, and retrieve indices are fixed in reusable
  templates.
- **Hardware execution optimization.** Candidate generation is batched, tree
  buffers remain at fixed GPU addresses, verification graphs are replayed with
  CUDA Graphs, and high-frequency Transformer paths are fused.

The released implementation additionally provides persistent K/V caches,
persistent decode workspaces, compact CPU-GPU control transfers, and graph-safe
fallback paths. These optimizations do not change the acceptance rule, Kalman
logic, or action definition.

## Main Results

### LIBERO simulation

The extended evaluation performs 50 trials for each task in all four LIBERO
task suites. KERV uses a fine-tuned OpenVLA verifier and a one-block LLaMA
drafter.

![KERV example on LIBERO-Goal](assets/kerv_simulation_example.png)

| Suite | Success Rate | Speedup | AFEP | Avg. Steps |
|---|---:|---:|---:|---:|
| LIBERO-Goal | 75.6% | 2.34x | 4.73 | 153.5 |
| LIBERO-Object | 72.3% | 2.22x | 4.71 | 186.8 |
| LIBERO-Spatial | 83.7% | **2.39x** | 4.67 | 120.9 |
| LIBERO-Long | 48.8% | 2.31x | 4.64 | 391.2 |

Speedup is measured against Naive VLA+SD in the corresponding suite. AFEP is
the average first error position. The paper reports up to **2.39x** simulation
speedup with minimal success-rate degradation.

### Runtime ablation on LIBERO-Goal

| Framework | Runtime configuration | Success Rate | Speedup |
|---|---|---:|---:|
| Naive VLA+SD | -- | 76.2% | 1.00x |
| SpecVLA (`r=9`) | -- | 75.4% | 1.19x |
| SpecVLA (`r=15`) | -- | 71.0% | 1.23x |
| KERV | + CPU-GPU deployment | 75.4% | 1.54x |
| KERV | + Static tree-based verification | 75.8% | 1.83x |
| KERV | + Hardware execution optimization | 75.4% | **2.34x** |

The hardware optimizations are lossless with respect to KERV inference
semantics; small success-rate variations arise from independent evaluation
runs.

### Real-world manipulation

KERV is also evaluated on an AgileX Piper robot arm in three task categories.

![KERV real-world manipulation example](assets/kerv_real_world_example.png)

| Task | Avg. Score | Speedup | AFEP | Avg. Steps |
|---|---:|---:|---:|---:|
| Atomic Grasping | 0.85 | 2.16x | 4.56 | 45.9 |
| Spatial Movement | 0.79 | **2.23x** | 4.17 | 81.2 |
| Long-Horizon Reasoning | 0.59 | 2.10x | 3.37 | 106.2 |

## Released Implementation

```text
KERV/
|-- run_kerv.py                    # public inference entry point
|-- KERV-RuntimeOptimization/
|   |-- KERVRuntimeOptimization/   # optimized operators and runtime support
|   |-- FlagScale/                 # launcher bridge, configuration, and license
|   `-- LICENSE                    # runtime-extension license
|-- openvla/                       # minimal OpenVLA/KERV runtime
|-- training/                      # drafter data generation and training
`-- docs/                          # operator and training documentation
```

The public implementation includes 18 registered embodied-inference operator
interfaces. The default accuracy-safe path enables the operators that passed
correctness and performance checks and retains native fallbacks for unsupported
layouts.

Key optimized paths include:

- fused Q/K/V projection;
- fused Gate-Up-SwiGLU execution;
- static-tree packing and tree attention;
- action verification and acceptance;
- RoPE with persistent K/V-cache writes;
- accepted-path K/V commit;
- action-space projection and selection;
- persistent CUDA Graph inputs, caches, and control buffers.

The operators are exposed through `torch.ops.flagos_embodied`; interface and
fallback details are documented in
[docs/OPERATORS.md](docs/OPERATORS.md). FlagScale is used by the released
launcher for configuration, process management, and reproducible execution,
while KERV remains independent of the launcher namespace and scheduling layer.

## Installation

The reference environment uses Linux, Python 3.10, CUDA 12.x, and BF16. Create
an isolated environment and install PyTorch for the CUDA version on your host:

```bash
git clone https://github.com/lusunn111/KERV.git KERV
cd KERV

conda create -n kerv python=3.10 -y
conda activate kerv

# Install a CUDA-compatible PyTorch build first.
python -m pip install -r requirements.txt
python -m pip install -e openvla
```

Install LIBERO and the runtime launcher:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
python -m pip install -e third_party/LIBERO

git clone https://github.com/flagos-ai/FlagScale.git third_party/FlagScale
python -m pip install -e third_party/FlagScale
```

FlashAttention is optional and depends on the local CUDA/PyTorch combination:

```bash
python -m pip install flash-attn --no-build-isolation
```

## Checkpoints

Weights are not included. Prepare a LIBERO-adapted OpenVLA verifier and a KERV
drafter checkpoint using the following layout:

```text
checkpoints/
|-- openvla-libero-goal/
|   |-- config.json
|   |-- dataset_statistics.json
|   |-- tokenizer.json
|   |-- preprocessor_config.json
|   `-- model-*.safetensors
`-- kerv-drafter/
    |-- config.json
    `-- pytorch_model.bin
```

The verifier follows the OpenVLA fine-tuning setup. The drafter is a single
LLaMA block trained from verifier features and action-token targets. See
[docs/TRAINING.md](docs/TRAINING.md) for data generation, DeepSpeed training,
and checkpoint export.

## Quick Start

Validate the complete configuration without loading model weights:

```bash
python run_kerv.py \
  --flagscale-root third_party/FlagScale \
  --dry-run
```

Run one LIBERO-Goal episode:

```bash
python run_kerv.py \
  --flagscale-root third_party/FlagScale \
  --base-checkpoint checkpoints/openvla-libero-goal \
  --draft-checkpoint checkpoints/kerv-drafter \
  --libero-config ~/.libero \
  --device 0
```

Run an eight-step smoke test:

```bash
python run_kerv.py \
  --flagscale-root third_party/FlagScale \
  --base-checkpoint checkpoints/openvla-libero-goal \
  --draft-checkpoint checkpoints/kerv-drafter \
  --libero-config ~/.libero \
  --max-episode-steps 8
```

Outputs are written to `outputs/kerv_libero_goal/`. Use `--background` to hand
the process to the launcher after configuration validation.

## Training the Drafter

The released training pipeline has two stages:

1. freeze the fine-tuned OpenVLA verifier and generate hidden-state/action-token
   supervision from LIBERO trajectories;
2. train the one-block drafter with DeepSpeed ZeRO-2 and export a standalone
   checkpoint.

The paper trains the drafter for approximately 12 hours on two NVIDIA A800
40 GB GPUs using human-egocentric trajectories from LIBERO. Generated samples
and checkpoints are excluded from version control. Reproduction commands and
configuration details are available in
[docs/TRAINING.md](docs/TRAINING.md).

## Reproducibility and Accuracy

For performance evaluation, report cold start separately from steady state;
the first Triton compilation and CUDA Graph capture must not be included in
steady-state speedup. When enabling a new optimized path, compare the following
values step by step against the BF16 reference path:

- drafter candidates and ordering;
- accepted length and best path;
- verifier loop count and next token;
- EOS and Kalman-compensation decisions;
- decoded robot action and final task result.

Experimental W8A16 and compact-tree modes are disabled by default. The public
default keeps the original KERV acceptance policy, Kalman logic, and action
semantics.

## Scope and Limitations

- Checkpoints, generated training data, datasets, and robot assets must be
  obtained or produced separately.
- The released default configuration targets batch-1 BF16 inference; operators
  use native fallbacks when a device or tensor layout is unsupported.
- The code is intended for research and should not be used directly in
  safety-critical robot control.
- Performance depends on the model checkpoint, task distribution, hardware,
  CUDA stack, and graph warm-up policy.

## Citation

If KERV is useful in your research, please cite the DAC paper:

```bibtex
@inproceedings{zheng2026kerv,
  title     = {{KERV}: Kinematic-Rectified Speculative Decoding for Embodied {VLA} Models},
  author    = {Zheng, Zihao and Mao, Zhihao and Li, Maoliang and Chen, Jiayu and Sun, Xinhao and Zhang, Zhaobo and Cao, Donggang and Mei, Hong and Chen, Xiang},
  booktitle = {Proceedings of the 63rd ACM/IEEE Design Automation Conference (DAC)},
  year      = {2026},
  note      = {To appear},
  eprint    = {2603.01581},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

The page range and DOI will be added when the final DAC bibliographic record
becomes available.

## Acknowledgements

This project builds on OpenVLA, LIBERO, EAGLE-2, DeepSpeed, PyTorch, Triton,
FlashAttention, FlagScale, and FlagGems. We thank their authors and open-source
communities. Third-party license information is provided in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

The repository contains MIT- and Apache-2.0-licensed components. See
[LICENSE](LICENSE),
[KERV-RuntimeOptimization/LICENSE](KERV-RuntimeOptimization/LICENSE),
[KERV-RuntimeOptimization/FlagScale/LICENSE](KERV-RuntimeOptimization/FlagScale/LICENSE),
and the license headers in imported components.
Models and datasets remain subject to their respective licenses.
