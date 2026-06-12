<h1 align="center">
  <img src="assets/VLA-JEPA.png" width="80" style="vertical-align: middle;" />
  <br/>
  VLA-JEPA
</h1>

<h3 align="center">
  <a href="https://arxiv.org/abs/2602.10098" style="color:#9C276A; text-decoration: none;">
    Enhancing Vision-Language-Action Models with a Latent World Model
  </a>
</h3>

<p align="center">
  <em>
    Jingwen Sun &middot; Wenyao Zhang &middot; Zekun Qi &middot; Shaojie Ren &middot; Zezhi Liu &middot; Hanxin Zhu &middot; Guangzhong Sun &middot; Xin Jin &middot; Zhibo Chen
  </em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.10098">
    <img src="https://img.shields.io/badge/📄_Paper-arXiv-b31b1b.svg" alt="Paper PDF">
  </a>
  &nbsp;
  <a href="https://ginwind.github.io/VLA-JEPA/">
    <img src="https://img.shields.io/badge/🌐_Project-Page-4285F4.svg" alt="Project Page">
  </a>
  &nbsp;
  <a href="https://huggingface.co/ginwind/VLA-JEPA">
    <img src="https://img.shields.io/badge/🤗_HuggingFace-Models-FFD21E.svg" alt="Hugging Face">
  </a>
  &nbsp;
  <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-228B22.svg" alt="Code License">
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB.svg" alt="Python 3.10+">
  &nbsp;
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg" alt="PyTorch 2.x">
</p>

<p align="center">
  ⭐ If this project is useful for your research, please consider giving us a star on GitHub!
</p>

---

<div align="center">
  <img src="assets/VLA-JEPA.png" width="90%" alt="VLA-JEPA Architecture Overview" />
  <br/>
  <sub><b>Figure 1.</b> VLA-JEPA integrates a latent world model (V-JEPA2) into a Vision-Language-Action framework, enabling future-aware action prediction through joint embedding predictive architecture.</sub>
</div>

---

## 📋 Table of Contents

- [✨ Highlights](#highlights)
- [🏗️ Architecture Overview](#architecture-overview)
- [📁 Project Structure](#project-structure)
- [🚧 TODO](#todo)
- [⚙️ Environment Setup](#environment-setup)
- [🔥 Training](#training)
  - [0️⃣ Pretrained Model Preparation](#pretrained-model-preparation)
  - [1️⃣ Data Preparation](#data-preparation)
  - [2️⃣ Start Training](#start-training)
  - [3️⃣ Optional: Custom Dataset Training](#optional-custom-dataset-training)
- [📊 Evaluation](#evaluation)
  - [LIBERO](#libero)
  - [LIBERO-Plus](#libero-plus)
  - [SimplerEnv](#simplerenv)
- [📈 Results & Benchmarks](#results--benchmarks)
- [💻 Hardware Requirements](#hardware-requirements)
- [❓ FAQ & Troubleshooting](#faq--troubleshooting)
- [🤝 Acknowledgement](#acknowledgement)
- [📝 Citation](#citation)

---

<a id="highlights"></a>
## ✨ Highlights

| Feature | Description |
|---|---|
| 🌍 **Latent World Model** | Integrates V-JEPA2 as an action-conditioned world model to predict future visual states in latent space, enabling physically grounded action generation. |
| 🧠 **Qwen3-VL Backbone** | Leverages the powerful Qwen3-VL-2B vision-language model for multimodal understanding and instruction-following. |
| 🎯 **DiT Action Head** | Uses a Diffusion Transformer (DiT-B) with flow-matching for precise continuous action prediction. |
| 🎬 **Video Co-Training** | Supports joint pre-training on human video data (e.g., SSV2) and robot manipulation data for better temporal dynamics understanding. |
| 🔧 **Modular Design** | Built on top of the starVLA framework with a registry-based, plug-and-play architecture for easy experimentation. |
| 📐 **Multi-View Support** | Natively handles multi-view camera inputs for richer spatial understanding. |

---

<a id="architecture-overview"></a>
## 🏗️ Architecture Overview

VLA-JEPA addresses a key limitation of standard VLA models — the lack of **future awareness**. By integrating a latent world model into the VLA pipeline, VLA-JEPA learns to predict future visual states, allowing the action policy to be informed by anticipated consequences.

### Core Components

```
┌──────────────────────────────────────────────────────────────────┐
│                       VLA-JEPA Framework                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │  Qwen3-VL   │───▶│  Latent Action    │───▶│  V-JEPA2       │  │
│  │  (VLM)      │    │  Tokens           │    │  (World Model) │  │
│  └─────────────┘    └──────────────────┘    └────────────────┘  │
│        │                                           │             │
│        │            ┌──────────────────┐           │             │
│        └───────────▶│  Embodied Action  │◀──────────┘             │
│                     │  Tokens           │                        │
│                     └────────┬─────────┘                        │
│                              │                                   │
│                     ┌────────▼─────────┐                        │
│                     │  DiT Action Head  │                        │
│                     │  (Flow Matching)  │                        │
│                     └────────┬─────────┘                        │
│                              │                                   │
│                     ┌────────▼─────────┐                        │
│                     │  Continuous       │                        │
│                     │  Action Output    │                        │
│                     └──────────────────┘                        │
└──────────────────────────────────────────────────────────────────┘
```

### Forward Pass Pipeline

1. **Multimodal Encoding** — Images and language instructions are processed by **Qwen3-VL-2B** to produce fused token embeddings with special action tokens injected.
2. **World Model Prediction** — Multi-view video frames are encoded by the frozen **V-JEPA2 encoder** (ViT-L). A lightweight predictor then takes encoded past frames + latent action tokens to predict future visual states (L1 loss against V-JEPA2 ground-truth embeddings).
3. **Action Decoding** — Embodied action tokens (conditioned on world model reasoning) are passed to a **DiT-B flow-matching action head** that generates precise continuous robot actions via iterative denoising.

### Training Objectives

| Loss | Description | Weight |
|------|-------------|--------|
| `action_loss` | Flow-matching diffusion loss on predicted action sequences | `1.0` |
| `wm_loss` | L1 teacher-forcing loss between predicted and actual V-JEPA2 future embeddings | `0.1` |
| `vlm_loss` | *(Pre-training only)* Language modeling loss for video understanding | `0.1` |

---

<a id="project-structure"></a>
## 📁 Project Structure

<details>
<summary>Click to expand full project tree</summary>

```
VLA-JEPA/
├── assets/                         # Images and figures for documentation
│   └── VLA-JEPA.png
├── deployment/                     # Model serving and deployment utilities
│   ├── model_server/               #   WebSocket-based model server
│   └── upload/                     #   HuggingFace upload scripts
├── examples/                       # Benchmark configs & evaluation scripts
│   ├── Droid/                      #   DROID dataset modality config
│   ├── LIBERO/                     #   LIBERO evaluation scripts
│   ├── LIBERO-Plus/                #   LIBERO-Plus evaluation scripts
│   └── SimplerEnv/                 #   SimplerEnv evaluation scripts & configs
├── scripts/                        # Training launch scripts
│   ├── config/
│   │   ├── vlajepa_cotrain.yaml    #     Pre-training config (robot + video)
│   │   └── vlajepa_robot_ft.yaml   #     Robot fine-tuning config
│   ├── vlajepa_cotrain.sh          #   Pre-training launch script
│   └── vlajepa_robot_ft.sh         #   Fine-tuning launch script
├── starVLA/                        # Core library
│   ├── config/                     #   DeepSpeed & training configs
│   │   ├── deepseeds/              #     DeepSpeed ZeRO configs
│   │   └── training/               #     Additional training configs
│   ├── dataloader/                 #   Dataset implementations
│   │   ├── gr00t_lerobot/          #     LeRobot v2.1 data adapter
│   │   ├── lerobot_datasets.py     #     Robot dataset loader
│   │   ├── video_datasets.py       #     Human video dataset loader
│   │   └── vlm_datasets.py         #     VLM pre-training dataset loader
│   ├── model/                      #   Model architectures
│   │   ├── framework/              #     Full model frameworks
│   │   │   ├── VLA_JEPA.py         #       ★ Main VLA-JEPA model
│   │   │   ├── base_framework.py   #       Base class for all frameworks
│   │   │   └── ...                 #       Other framework variants
│   │   ├── modules/                #     Sub-modules
│   │   │   ├── action_model/       #       DiT / Flow-matching action heads
│   │   │   ├── world_model/        #       V-JEPA2 predictor modules
│   │   │   ├── vlm/                #       Qwen3-VL interface
│   │   │   └── projector/          #       Feature projection layers
│   │   └── tools.py                #     Framework registry utilities
│   └── training/                   #   Training loops & utilities
│       ├── train_vlajepa_cotrain.py #     Co-training script (video + robot)
│       ├── train_vlajepa_video.py   #     Video-only pre-training
│       └── trainer_utils/          #     Logging, scheduling, etc.
├── pyproject.toml                  # Package configuration
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

</details>

---

<a id="todo"></a>
## 🚧 TODO

- [x] Partial training code
- [x] LIBERO evaluation code
- [x] LIBERO-Plus evaluation code
- [x] SimplerEnv evaluation code
- [x] Training codes for custom datasets
- [ ] Real-world deployment guide
- [ ] Detailed training recipes & hyperparameter tuning guide

---

<a id="environment-setup"></a>
## ⚙️ Environment Setup

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with ≥24 GB VRAM (A100/H100 recommended for training)
- **CUDA**: 12.1+
- **Python**: 3.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/ginwind/VLA-JEPA
cd VLA-JEPA

# Create and activate conda environment
conda create -n VLA_JEPA python=3.10 -y
conda activate VLA_JEPA

# Install dependencies
pip install -r requirements.txt

# Install FlashAttention2 (required for efficient attention)
pip install flash-attn --no-build-isolation

# Install the project in editable mode
pip install -e .
```

> **Note**: This repository's code is built on top of [starVLA](https://github.com/starVLA/starVLA). The `starVLA` package is included directly in this repository — no separate installation is needed.

---

<a id="training"></a>
## 🔥 Training

VLA-JEPA supports two training modes:

| Mode | Script | Config | Description |
|------|--------|--------|-------------|
| **Co-Training** (Pre-training) | `vlajepa_cotrain.sh` | `vlajepa_cotrain.yaml` | Joint training on robot manipulation data + human video data |
| **Robot Fine-Tuning** (Post-training) | `vlajepa_robot_ft.sh` | `vlajepa_robot_ft.yaml` | Fine-tuning on specific robot benchmark data |

<a id="pretrained-model-preparation"></a>
### 0️⃣ Pretrained Model Preparation

Download the following pretrained checkpoints:

| Model | Source | Purpose |
|-------|--------|---------|
| **Qwen3-VL-2B-Instruct** | [🤗 HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Vision-Language backbone |
| **V-JEPA2 ViT-L** | [🤗 HuggingFace](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | Frozen world model encoder |

<a id="data-preparation"></a>
### 1️⃣ Data Preparation

Download the following datasets:

| Dataset | Type | Link |
|---------|------|------|
| **SSV2** (Something-Something v2) | Human Video | [🤗 HuggingFace](https://huggingface.co/datasets/morpheushoc/something-something-v2) |
| **DROID** | Robot Manipulation | [🤗 HuggingFace](https://huggingface.co/datasets/IPEC-COMMUNITY/droid_lerobot) |
| **LIBERO** | Robot Manipulation | [🤗 HuggingFace](https://huggingface.co/collections/IPEC-COMMUNITY/libero-benchmark-dataset) |
| **BridgeV2** | Robot Manipulation | [🤗 HuggingFace](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) |
| **Fractal** | Robot Manipulation | [🤗 HuggingFace](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) |

> **Important**: For robot datasets, you need to add a `modality.json` file under the `meta/` subdirectory of each LeRobot dataset. Pre-made `modality.json` files are provided:
> - `./examples/Droid/` — for DROID
> - `./examples/LIBERO/` — for LIBERO
> - `./examples/SimplerEnv/` — for BridgeV2 and Fractal

<a id="start-training"></a>
### 2️⃣ Start Training

1. **Select the appropriate script** from the [`/scripts`](./scripts) directory based on your training mode.

2. **Update the YAML configuration file** with your local paths:

   ```yaml
   # Model checkpoints
   framework:
     qwenvl:
       base_vlm: /path/to/Qwen3-VL-2B-Instruct
     vj2_model:
       base_encoder: /path/to/vjepa2-vitl-fpc64-256

   # Dataset paths
   datasets:
     vla_data:
       data_root_dir: /path/to/robot/dataset
     video_data:
       video_dir: /path/to/video/dataset
       text_file: /path/to/video/annotations.csv
   ```

3. **Launch training**:

   ```bash
   # Co-training (pre-training with video + robot data)
   bash scripts/vlajepa_cotrain.sh

   # Robot fine-tuning (post-training on specific benchmark)
   bash scripts/vlajepa_robot_ft.sh
   ```

<details>
<summary>📝 Key training hyperparameters (click to expand)</summary>

| Parameter | Co-Training | Fine-Tuning |
|-----------|-------------|-------------|
| Max training steps | 50,000 | — |
| Warmup steps | 5,000 | — |
| Base learning rate | 3e-5 | — |
| VLM learning rate | 1e-5 | — |
| Action model LR | 1e-4 | — |
| LR scheduler | Cosine with min LR | — |
| Batch size (per device) | 16 | — |
| Gradient accumulation | 1 | — |
| Mixed precision | BF16 | BF16 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) | — |
| Max gradient norm | 1.0 | — |
| Diffusion steps (training) | 8 (repeated) | — |
| Diffusion steps (inference) | 4 | 4 |

</details>

<a id="optional-custom-dataset-training"></a>
### 3️⃣ Optional: Custom Dataset Training

VLA-JEPA supports training on both robot datasets and human video datasets. You can run custom training by specifying robot data and/or human videos in your configuration.

<details>
<summary>🤖 <b>Robot Data</b> — Using custom robot datasets</summary>

We support training with datasets in the **LeRobot v2.1** format. Convert your custom robot dataset to LeRobot v2.1 first, then:

1. **Define a data config class** in [`data_config.py`](./starVLA/dataloader/gr00t_lerobot/data_config.py):
   - The video-key fields should match values predefined in your `modality.json` (see [`modality.json`](./examples/Droid/modality.json) for reference).

2. **Register your dataset** by adding a mapping from `robot_type` to the config class in `ROBOT_TYPE_CONFIG_MAP`.

3. **Configure the data mixture** in [`mixtures.py`](./starVLA/dataloader/gr00t_lerobot/mixtures.py):
   - The dict key in `DATASET_NAMED_MIXTURES` corresponds to `datasets.vla_data.data_mix` in the YAML config.
   - Each sub-dataset tuple contains: `(subdirectory, version, robot_type)`.

4. **Update the YAML config** and launch training.

</details>

<details>
<summary>🎬 <b>Human Video</b> — Using custom video datasets</summary>

**Option A — Use the built-in video dataloader**: Configure the `datasets.video_data` section in your YAML:

```yaml
datasets:
  video_data:
    dataset_py: video_datasets          # Built-in loader (no change needed)
    video_dir: /path/to/videos          # Directory containing video files
    text_file: /path/to/annotations.csv # Headerless CSV: index, description
    CoT_prompt: "Your task is {instruction}. Infer the temporal dynamics of future frames {actions}."
    extensions: [webm]                  # Video file extensions
    resolution_size: 224
    video_resolution_size: 256
    per_device_batch_size: 16
```

**Option B — Custom DataLoader**: Implement your own DataLoader and register it in `build_dataloader` within [`./starVLA/dataloader/__init__.py`](./starVLA/dataloader/__init__.py).

</details>

---

<a id="evaluation"></a>
## 📊 Evaluation

Download the model checkpoints from Hugging Face: **[ginwind/VLA-JEPA](https://huggingface.co/ginwind/VLA-JEPA)**

**Additional dependencies** (install into your `VLA_JEPA` environment):
```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

---

<a id="libero"></a>
### LIBERO

<details open>
<summary>Expand instructions</summary>

1. **Setup**: Prepare the LIBERO benchmark in a **separate conda environment** following the [official LIBERO instructions](https://github.com/Lifelong-Robot-Learning/LIBERO).

2. **Configuration**: In the downloaded checkpoint folder, update `config.json` and `config.yaml`:
   - `framework.qwenvl.basevlm` → path to Qwen3-VL-2B checkpoint
   - `framework.vj2_model.base_encoder` → path to V-JEPA2 encoder checkpoint

3. **Edit evaluation script** ([`examples/LIBERO/eval_libero.sh`](./examples/LIBERO/eval_libero.sh)):
   - Line 4: Set `LIBERO_HOME` to your local LIBERO code path
   - Line 9: Set `sim_python` to the LIBERO conda Python executable
   - Line 11: Set `your_ckpt` to `LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`

4. **Run** (evaluates 4 task suites in parallel across 4 GPUs):
   ```bash
   bash ./examples/LIBERO/eval_libero.sh
   ```

</details>

---

<a id="libero-plus"></a>
### LIBERO-Plus

<details>
<summary>Expand instructions</summary>

1. **Setup**: Clone [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus). Update line 121 in [`./examples/LIBERO-Plus/libero_plus_init.py`](./examples/LIBERO-Plus/libero_plus_init.py) to point to your `LIBERO-Plus/libero/libero/benchmark/task_classification.json`. Replace the original `__init__.py` with the provided modified implementation to enable evaluation over perturbation dimensions. Build the benchmark in a separate conda environment.

2. **Configuration**: Same as LIBERO — update `config.json` and `config.yaml` with local checkpoint paths.

3. **Edit evaluation script** ([`examples/LIBERO-Plus/eval_libero_plus.sh`](./examples/LIBERO-Plus/eval_libero_plus.sh)):
   - Line 4: Set `LIBERO_HOME` to your local LIBERO-Plus code path
   - Line 9: Set `sim_python` to the LIBERO-Plus conda Python executable
   - Line 11: Set `your_ckpt` to `LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`

4. **Run** (evaluates 7 perturbation dimensions in parallel across 7 GPUs):
   ```bash
   bash ./examples/LIBERO-Plus/eval_libero_plus.sh
   ```

</details>

---

<a id="simplerenv"></a>
### SimplerEnv

<details>
<summary>Expand instructions</summary>

1. **Setup**: Clone [SimplerEnv](https://github.com/simpler-env/SimplerEnv) and follow the official installation instructions in a separate conda environment.

2. **Configuration**: Update `config.json` and `config.yaml` with local checkpoint paths (same fields as LIBERO).

3. **Edit evaluation script** ([`examples/SimplerEnv/eval_files/auto_eval_scripts/batch_evaluate.sh`](examples/SimplerEnv/eval_files/auto_eval_scripts/batch_evaluate.sh)):
   - Set `SimplerEnv_PATH` to your local SimplerEnv code path
   - Set `sim_python` to the SimplerEnv conda Python executable
   - Set `MODEL_PATH` to `SimplerEnv/checkpoints/VLA-JEPA-Simpler.pt`

4. **Run evaluation**:
   ```bash
   bash examples/SimplerEnv/eval_files/auto_eval_scripts/batch_evaluate.sh
   ```

5. **Compute success rates** (after evaluation generates rollout videos):
   ```bash
   # <task_suite>: pick_coke_can | move_near | drawer | long_horizon_apple_in_drawer | bridge_put_on
   # Note: bridge_put_on = WidowX robot; others = Google Robot
   bash ./examples/SimplerEnv/eval_files/auto_eval_scripts/calc_success_rate.sh <task_suite> <model_path> <log_dir>
   ```

</details>

---

> **⚠️ Important Notes for All Evaluations**:
> - Ensure each process has access to a dedicated GPU.
> - Verify all checkpoint paths in configuration files before running.
> - LIBERO: 4 GPUs (4 task suites in parallel)
> - LIBERO-Plus & SimplerEnv: 8 GPUs recommended (parallel evaluation)
> - If you have fewer GPUs, modify the parallelization logic in the launch scripts accordingly.

---

<a id="results--benchmarks"></a>
## 📈 Results & Benchmarks

### LIBERO Benchmark

> Results reported from the paper ([arXiv:2602.10098](https://arxiv.org/abs/2602.10098)).

| Method | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | **Average** |
|--------|:-:|:-:|:-:|:-:|:-:|
| OpenVLA | 62.0 | 73.0 | 57.0 | 38.5 | 57.6 |
| π₀ (fine-tuned) | 72.0 | 80.0 | 67.0 | 48.5 | 66.9 |
| **VLA-JEPA (Ours)** | **86.5** | **89.5** | **83.5** | **68.0** | **81.9** |

### SimplerEnv (Simulated Google Robot & WidowX)

*Refer to the [paper](https://arxiv.org/abs/2602.10098) for full SimplerEnv and LIBERO-Plus results across all perturbation dimensions.*

---

<a id="hardware-requirements"></a>
## 💻 Hardware Requirements

| Use Case | GPUs | VRAM per GPU | Notes |
|----------|------|-------------|-------|
| **Inference** | 1 | ≥ 24 GB | Single-GPU inference with BF16 |
| **Fine-tuning** | 4–8 | ≥ 40 GB | DeepSpeed ZeRO-2, 8× recommended |
| **Co-training** (Pre-training) | 8 | ≥ 80 GB | A100/H100 recommended |
| **LIBERO Evaluation** | 4 | ≥ 24 GB | 4 task suites in parallel |
| **SimplerEnv Evaluation** | 8 | ≥ 24 GB | Parallel sub-task evaluation |

---

<a id="faq--troubleshooting"></a>
## ❓ FAQ & Troubleshooting

<details>
<summary><b>Q: CUDA out of memory during training</b></summary>

- Reduce `per_device_batch_size` in your YAML config
- Enable gradient checkpointing: `enable_gradient_checkpointing: true`
- Use DeepSpeed ZeRO-3 instead of ZeRO-2
- Reduce `repeated_diffusion_steps` (default: 8)

</details>

<details>
<summary><b>Q: FlashAttention2 installation fails</b></summary>

- Ensure you have CUDA 12.1+ and a compatible GPU (Ampere or later)
- Try: `pip install flash-attn --no-build-isolation --no-cache-dir`
- Check [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention) for build requirements

</details>

<details>
<summary><b>Q: NCCL timeout errors during multi-GPU training</b></summary>

- Increase the timeout: `export NCCL_TIMEOUT=3600`
- Ensure all GPUs are visible: `nvidia-smi`
- Check network interface: `export NCCL_SOCKET_IFNAME=eth0` (adjust to your interface)
- For InfiniBand issues: `export NCCL_IB_DISABLE=1`

</details>

<details>
<summary><b>Q: How do I train on a single GPU?</b></summary>

- Modify the launch script: set `--num_processes 1`
- Reduce batch size and enable gradient accumulation
- This is primarily for debugging — multi-GPU is strongly recommended for full training

</details>

<details>
<summary><b>Q: Can I use a different VLM backbone instead of Qwen3-VL?</b></summary>

The framework is modular — you can implement a new VLM interface under `starVLA/model/modules/vlm/` and register it. However, the current action token injection and prompt formatting logic is tightly coupled with Qwen3-VL's tokenizer, so adaptation will require changes in the framework class.

</details>

---

<a id="acknowledgement"></a>
## 🤝 Acknowledgement

We extend our sincere gratitude to the following projects for their invaluable open-source contributions:

- [**starVLA**](https://github.com/starVLA/starVLA) — The foundational codebase and modular framework
- [**V-JEPA2**](https://github.com/facebookresearch/vjepa2) — The world model encoder (Meta AI)
- [**Qwen3-VL**](https://github.com/QwenLM/Qwen2.5-VL) — The vision-language backbone (Alibaba)
- [**LeRobot**](https://github.com/huggingface/lerobot) — Standardized robot dataset format (Hugging Face)
- [**GR00T**](https://developer.nvidia.com/isaac/groot) — Flow-matching action head architecture inspiration (NVIDIA)

---

<a id="citation"></a>
## 📝 Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2602.10098):

```bibtex
@misc{vlajepa2026,
    title     = {VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model},
    author    = {Jingwen Sun and Wenyao Zhang and Zekun Qi and Shaojie Ren and
                 Zezhi Liu and Hanxin Zhu and Guangzhong Sun and Xin Jin and Zhibo Chen},
    year      = {2026},
    eprint    = {2602.10098},
    archivePrefix = {arXiv},
    primaryClass  = {cs.RO},
    url       = {https://arxiv.org/abs/2602.10098},
}
```

---

<p align="center">
  Made with ❤️ by the VLA-JEPA Team
</p>
