<div align="center">
VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model
</div>

## TODO
- [✅] Release the training code
- [❌] Release the evaluation code for LIBERO
- [❌] Release the evaluation code for LIBERO-Plus
- [❌] Release the evaluation code for SimplerEnv

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

## Acknowledgement

We extend our sincere gratitude to the [starVLA](https://scholar.google.com/citations?user=leXXHKwAAAAJ&hl=zh-CN) project and the [V-JEPA2](https://github.com/facebookresearch/vjepa2) project for their invaluable open-source contributions.
