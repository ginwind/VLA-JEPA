<div align="center">
VLA-JEPA: Enhancing Vision-language-action model with Latent World Model
</div>

## TODO
- [✅] Release the training code
- [❌] Release the evaluation code of LIBERO
- [❌] Release the evaluation code of LIBERO-Plus
- [❌] Release the evaluation code of SimplerEnv

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

## Acknowledgement

We would like to express our deepest gratitude to [starVLA](https://scholar.google.com/citations?user=leXXHKwAAAAJ&hl=zh-CN) project and [V-JEPA2](https://github.com/facebookresearch/vjepa2) project for their inspiring open-source contributions.
