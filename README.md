<h3 align="center" style="font-size:48px; font-weight:bold; color:#9C276A; margin: 0;">
  <a href="https://arxiv.org/abs/2602.10098" style="color:#9C276A; text-decoration: none;">
    VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model
  </a>
</h3>

<div align="center">
<p>
  <a href="https://arxiv.org/abs/2602.10098">
    <img src="https://img.shields.io/badge/Paper-PDF-orange.svg" alt="Paper PDF">
  </a>
  <a href="https://ginwind.github.io/VLA-JEPA/">
    <img src="https://img.shields.io/badge/Project-Page-Green.svg" alt="Project Page">
  </a>
  <a href="https://huggingface.co/ginwind/VLA-JEPA">
    <img src="https://img.shields.io/badge/🤗-Hugging_Face-yellow.svg" alt="Hugging Face">
  </a>
  <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg" alt="Code License">
  </a>
</p>
<p align="center">
  ⭐ If our project helps you, please give us a star on GitHub to support us!
</p>
</div>

## TODO
- [x] Partial training code
- [x] LIBERO evaluation code
- [ ] LIBERO-Plus evaluation code
- [ ] SimplerEnv evaluation code
- [ ] Training codes for custom datasets

## Environment Setup

```
git clone https://github.com/ginwind/VLA-JEPA

# Create conda environment
conda create -n VLA_JEPA python=3.10 -y
conda activate VLA_JEPA

# Install requirements
pip install -r requirements.txt

# Install FlashAttention2
pip install flash-attn --no-build-isolation

# Install project
pip install -e .
```

This repository's code is based on the [starVLA](https://github.com/starVLA/starVLA).

## Training

### 0️⃣ Pretrained Model Preparation
Download the [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and the [V-JEPA2 encoder](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256).  

### 1️⃣ Data Preparation

Download the following datasets:

- [ssv2](https://huggingface.co/datasets/HuggingFaceM4/something_something_v2)
- [Droid](https://huggingface.co/datasets/IPEC-COMMUNITY/droid_lerobot)
- [LIBERO](https://huggingface.co/collections/IPEC-COMMUNITY/libero-benchmark-dataset)
- [BridgeV2](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot)
- [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot)

### 2️⃣ Start Training
Depending on whether you are conducting pre-training or post-training, select the appropriate training script and YAML configuration file from the [`/scripts`](./scripts) directory.

Ensure the following configurations are updated in the YAML file:
- `framework.qwenvl.basevlm` and `framework.vj2_model.base_encoder` should be set to the paths of your respective checkpoints.
- Update `datasets.vla_data.data_root_dir`, `datasets.video_data.video_dir`, and `datasets.video_data.text_file` to match the paths of your datasets.

Once the configurations are updated, you can proceed to start the training process.

## Evaluation

Download the model checkpoints from Hugging Face: https://huggingface.co/ginwind/VLA-JEPA

### LIBERO

- **Environment:** Install the required Python packages into your `VLA-JEPA` environment:
```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

- **LIBERO setup:** Prepare the LIBERO benchmark in a separate conda environment following the official LIBERO instructions: https://github.com/Lifelong-Robot-Learning/LIBERO

- **Configuration:** In the downloaded checkpoint folder, update `config.json` and `config.yaml` to point the following fields to your local checkpoints:
  - `framework.qwenvl.basevlm`: path to the Qwen3-VL-2B checkpoint
  - `framework.vj2_model.base_encoder`: path to the V-JEPA encoder checkpoint

- **Evaluation script:** Edit [`examples/LIBERO/eval_libero.sh`](./examples/LIBERO/eval_libero.sh) and set the `LIBERO_HOME` environment variable (line 4) to your local LIBERO code path, and set the `sim_python` variable (line 9) to the Python executable of the LIBERO conda environment. Finally, set the `your_ckpt` variable (line 11) to the path of the downloaded `LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`.

- **Run evaluation:** Launch the evaluation (the script runs the four task suites in parallel across 4 GPUs):
```bash
bash ./examples/LIBERO/eval_libero.sh
```

- **Notes:** Ensure each process has access to a GPU and verify that all checkpoint paths in the configuration files are correct before running the evaluation.


## Acknowledgement

We extend our sincere gratitude to the [starVLA](https://scholar.google.com/citations?user=leXXHKwAAAAJ&hl=zh-CN) project and the [V-JEPA2](https://github.com/facebookresearch/vjepa2) project for their invaluable open-source contributions.
