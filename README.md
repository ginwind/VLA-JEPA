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

<div align="center">
  <img src="assets/VLA-JEPA.png" width="90%" alt="VLA-JEPA overview" />
</div>

<a id="table-of-contents"></a>
## Table of Contents
- [Table of Contents](#table-of-contents)
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
- [🤝 Acknowledgement](#acknowledgement)
- [📝 Citation](#citation)
  
<a id="todo"></a>
## 🚧 TODO
- [x] Partial training code
- [x] LIBERO evaluation code
- [x] LIBERO-Plus evaluation code
- [x] SimplerEnv evaluation code
- [x] Training codes for custom datasets

<a id="environment-setup"></a>
## ⚙️ Environment Setup

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

<a id="training"></a>
## 🔥 Training

<a id="pretrained-model-preparation"></a>
### 0️⃣ Pretrained Model Preparation
Download the [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and the [V-JEPA2 encoder](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256).  

<a id="data-preparation"></a>
### 1️⃣ Data Preparation

Download the following datasets:

- [ssv2](https://huggingface.co/datasets/morpheushoc/something-something-v2)
- [Droid](https://huggingface.co/datasets/IPEC-COMMUNITY/droid_lerobot)
- [LIBERO](https://huggingface.co/collections/IPEC-COMMUNITY/libero-benchmark-dataset)
- [BridgeV2](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot)
- [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot)

For robot datasets, you need to add a `modality.json` file under the `meta/` subdirectory of each LeRobot dataset. The `modality.json` files for LIBERO, BridgeV2, Fractal, and Droid are provided under `./examples` (BridgeV2 and Fractal are under `./examples/SimplerEnv`).

<a id="start-training"></a>
### 2️⃣ Start Training
Depending on whether you are conducting pre-training or post-training, select the appropriate training script and YAML configuration file from the [`/scripts`](./scripts) directory.

Ensure the following configurations are updated in the YAML file:
- `framework.qwenvl.basevlm` and `framework.vj2_model.base_encoder` should be set to the paths of your respective checkpoints.
- Update `datasets.vla_data.data_root_dir`, `datasets.video_data.video_dir`, and `datasets.video_data.text_file` to match the paths of your datasets.

Once the configurations are updated, you can proceed to start the training process.

<a id="optional-custom-dataset-training"></a>
### 3️⃣ Optional: Custom Dataset Training
VLA-JEPA supports training on both robot datasets and human video datasets. You can run custom training by specifying robot data and/or human videos in your configuration.

- **Robot Data:** We support training with datasets in the LeRobot v2.1 format. Convert your custom robot dataset to LeRobot v2.1 first.
  - Define a custom robot dataset config class in [`data_config.py`](./starVLA/dataloader/gr00t_lerobot/data_config.py) (its video-key fields should match the values predefined in `modality.json`; see [`modality.json`](./examples/Droid/modality.json)), and add a mapping from `robot_type` to the config class in `ROBOT_TYPE_CONFIG_MAP`.
  - `robot_type` is specified by `DATASET_NAMED_MIXTURES` in [`mixtures.py`](./starVLA/dataloader/gr00t_lerobot/mixtures.py): the dict key corresponds to `datasets.vla_data.data_mix` in the YAML training config, and the value is a tuple of sub-datasets. Each sub-dataset tuple contains three items: subdirectory, version, and `robot_type`. The `robot_type` selects the corresponding config for state/action normalization and other field alignment.
  - Finally, update the YAML config accordingly and launch training.

- **Human Video:** You can implement your own DataLoader and update the mapping from `dataset_py` to a dataloader in `build_dataloader` within [`./starVLA/dataloader/__init__.py`](./starVLA/dataloader/__init__.py). Alternatively, use our video dataloader and configure `datasets.video_data` in the YAML file:
  - dataset_py: use our video dataloader (no change needed)
  - video_dir: directory that contains video files; each file is named by its `index`, and the suffix is controlled by `extensions`
  - text_file: a headerless CSV where the first column is `index` and the second column is the video text description
  - CoT_prompt: prompt template for latent-action training (no change needed)
  - extensions: list of video file extensions



<a id="evaluation"></a>
## 📊 Evaluation

Download the model checkpoints from Hugging Face: https://huggingface.co/ginwind/VLA-JEPA

**Environment:** Install the required Python packages into your `VLA-JEPA` environment:
```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

<a id="libero"></a>
### LIBERO

- **LIBERO setup:** Prepare the LIBERO benchmark in a separate conda environment following the official LIBERO instructions: https://github.com/Lifelong-Robot-Learning/LIBERO

- **Configuration:** In the downloaded checkpoint folder, update `config.json` and `config.yaml` to point the following fields to your local checkpoints:
  - `framework.qwenvl.basevlm`: path to the Qwen3-VL-2B checkpoint
  - `framework.vj2_model.base_encoder`: path to the V-JEPA encoder checkpoint

- **Evaluation script:** Edit [`examples/LIBERO/eval_libero.sh`](./examples/LIBERO/eval_libero.sh) and set the `LIBERO_HOME` environment variable (line 4) to your local LIBERO code path, and set the `sim_python` variable (line 9) to the Python executable of the LIBERO conda environment. Finally, set the `your_ckpt` variable (line 11) to the path of the downloaded `LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`.

- **Run evaluation:** Launch the evaluation (the script runs the four task suites in parallel across 4 GPUs):
```bash
bash ./examples/LIBERO/eval_libero.sh
```

<a id="libero-plus"></a>
### LIBERO-Plus


- **LIBERO-Plus setup:** Clone the LIBERO-Plus repository: https://github.com/sylvestf/LIBERO-plus. In [`./examples/LIBERO-Plus/libero_plus_init.py`](./examples/LIBERO-Plus/libero_plus_init.py), update line 121 to point to your `LIBERO-Plus/libero/libero/benchmark/task_classification.json`. Replace the original `LIBERO-Plus/libero/libero/benchmark/__init__.py` with the provided modified implementation (see [`./examples/LIBERO-Plus/libero_plus_init.py`](./examples/LIBERO-Plus/libero_plus_init.py)) to enable evaluation over perturbation dimensions. Finally, follow the official LIBERO-Plus installation instructions and build the benchmark in a separate conda environment.

- **Configuration:** In the downloaded checkpoint folder, update `config.json` and `config.yaml` to point the following fields to your local checkpoints:
  - `framework.qwenvl.basevlm`: path to the Qwen3-VL-2B checkpoint
  - `framework.vj2_model.base_encoder`: path to the V-JEPA encoder checkpoint

- **Evaluation script:** Edit [`examples/LIBERO-Plus/eval_libero_plus.sh`](./examples/LIBERO-Plus/eval_libero_plus.sh) and set the `LIBERO_HOME` environment variable (line 4) to your local LIBERO-Plus code path, and set the `sim_python` variable (line 9) to the Python executable of the LIBERO-Plus conda environment. Finally, set the `your_ckpt` variable (line 11) to the path of the downloaded `LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`.

- **Run evaluation:** Launch the evaluation (the script runs the seven pertubation dimensions in parallel across 7 GPUs):
```bash
bash ./examples/LIBERO-Plus/eval_libero_plus.sh
```

<a id="simplerenv"></a>
### SimplerEnv

- **SimplerEnv setup:** Clone the SimplerEnv repository: https://github.com/simpler-env/SimplerEnv and follow the official SimplerEnv installation instructions and build the benchmark in a separate conda environment.

- **Configuration:** In the downloaded checkpoint folder, update `config.json` and `config.yaml` to point the following fields to your local checkpoints:
  - `framework.qwenvl.basevlm`: path to the Qwen3-VL-2B checkpoint
  - `framework.vj2_model.base_encoder`: path to the V-JEPA encoder checkpoint

- **Evaluation script:** Edit [`examples/SimplerEnv/eval_files/auto_eval_scripts/batch_evaluate.sh`](examples/SimplerEnv/eval_files/auto_eval_scripts/batch_evaluate.sh) and set the `SimplerEnv_PATH` environment variable to your local SimplerEnv code path, and set the `sim_python` variable to the Python executable of the SimplerEnv conda environment. Finally, set the `MODEL_PATH` variable to the path of the downloaded `SimplerEnv/checkpoints/VLA-JEPA-Simpler.pt`.

- **Run evaluation:** Launch the evaluation:
```bash
bash examples/SimplerEnv/eval_files/auto_eval_scripts/batch_evaluate.sh
```

- **Compute success rates:** After the previous step, SimplerEnv will generate evaluation rollout videos for each sub-task. You can then compute task success rates with [`examples/SimplerEnv/eval_files/auto_eval_scripts/calc_success_rate.sh`](examples/SimplerEnv/eval_files/auto_eval_scripts/calc_success_rate.sh) as follows:
```bash
# <task_suite> must be one of: pick_coke_can | move_near | drawer | long_horizon_apple_in_drawer | bridge_put_on.
# Note: bridge_put_on corresponds to the WidowX robot evaluation; the other four correspond to the Google Robot evaluation.
# <model_path> is the path to `VLA-JEPA-Simpler.pt`, and <log_dir> is the root directory that contains the generated videos
# (by default, this is saved under `./results` within the evaluation output directory).
bash ./examples/SimplerEnv/eval_files/auto_eval_scripts/calc_success_rate.sh <task_suite> <model_path> <log_dir>
```

**Notes:** Ensure each process has access to a GPU and verify that all checkpoint paths in the configuration files are correct before running the evaluation. For LIBERO, we evaluate the 4 task suites in parallel on 4 GPUs. For LIBERO-Plus and SimplerEnv, we run evaluations in parallel on 8 GPUs. If you have fewer GPUs available, modify the parallelization logic in the launch scripts accordingly.


<a id="acknowledgement"></a>
## 🤝 Acknowledgement

We extend our sincere gratitude to the [starVLA](https://github.com/starVLA/starVLA) project and the [V-JEPA2](https://github.com/facebookresearch/vjepa2) project for their invaluable open-source contributions.

<a id="citation"></a>
## 📝 Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2602.10098):
```
@misc{vlajepa2026,
          title={VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model}, 
          author={Jingwen Sun and Wenyao Zhang and Zekun Qi and Shaojie Ren and Zezhi Liu and Hanxin Zhu and Guangzhong Sun and Xin Jin and Zhibo Chen},
          year={2026},
          eprint={2602.10098},
          archivePrefix={arXiv},
          primaryClass={cs.RO},
          url={https://arxiv.org/abs/2602.10098}, 
    }
```
