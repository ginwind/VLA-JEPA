"""
Trainer utility helpers.

This module contains small helpers for CLI overrides, parameter-group construction,
distributed-safe printing, model freezing/loading, and a few evaluation utilities.
"""

import json
import os
import re
from functools import wraps
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from PIL import Image
from torchvision.ops import box_iou

logger = get_logger(__name__)


def _distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_rank_zero() -> bool:
    return not _distributed_ready() or dist.get_rank() == 0


def _barrier_if_distributed() -> None:
    if _distributed_ready():
        dist.barrier()


def _resolve_module(model, module_path: str):
    module = model
    for attr in module_path.split("."):
        module = getattr(module, attr)
    return module


def normalize_dotlist_args(args):
    """
    Convert CLI args like ['--x.y', 'val'] and ['--flag'] into OmegaConf dotlist entries.
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")

    return normalized


def build_param_lr_groups(model, cfg):
    """
    Build optimizer parameter groups from cfg.trainer.learning_rate.

    Supports per-module learning rates and excludes parameters belonging to
    cfg.trainer.freeze_modules.
    """
    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)

    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    frozen_params = set()
    param_groups = []

    for freeze_path in freeze_patterns:
        try:
            module = _resolve_module(model, freeze_path)
            frozen_params.update(id(p) for p in module.parameters())
        except AttributeError:
            logger.warning(f"Freeze module path does not exist: {freeze_path}")

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue

        try:
            module = _resolve_module(model, module_name)
        except AttributeError:
            logger.warning(f"Module path `{module_name}` not found in model; skipping custom learning rate.")
            continue

        params = [p for p in module.parameters() if id(p) not in frozen_params]
        if params:
            param_groups.append({"params": params, "lr": lr, "name": module_name})
            used_params.update(id(p) for p in params)

    other_params = [p for p in model.parameters() if id(p) not in used_params and id(p) not in frozen_params]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


def only_main_process(func):
    """Decorator: only run the wrapped function on rank 0."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _is_rank_zero():
            return None
        return func(*args, **kwargs)

    return wrapper


def resize_images(images, target_size=(224, 224)):
    """
    Recursively resize all PIL images in a nested list.

    Args:
        images: Nested list of PIL images or a single PIL image.
        target_size: Target size as (width, height).

    Returns:
        Resized images with the original nesting structure preserved.
    """
    if isinstance(images, Image.Image):
        return images.resize(target_size)
    if isinstance(images, list):
        return [resize_images(img, target_size) for img in images]
    raise ValueError("Unsupported image type or structure.")


class TrainerUtils:
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """
        Freeze submodules from a comma-separated relative module path list.
        """
        frozen = []
        if freeze_modules and isinstance(freeze_modules, str):
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

            for path in patterns:
                try:
                    module = _resolve_module(model, path)
                except AttributeError:
                    logger.warning(f"Module path does not exist, cannot freeze: {path}")
                    continue

                for param in module.parameters():
                    param.requires_grad = False
                frozen.append(path)

        _barrier_if_distributed()
        if _is_rank_zero():
            print(f"Frozen modules: {frozen}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        Print the total and trainable parameter counts for the model.
        """
        if not _is_rank_zero():
            return None

        print("model parameter statistics:")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
        )
        return num_params, num_trainable_params

    @staticmethod
    def load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None):
        """
        Load a checkpoint into the full model or selected submodules.
        """
        if not checkpoint_path:
            return model

        if _is_rank_zero():
            print(f"Loading checkpoint: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Loading checkpoint failed: {e}") from e

        if reload_modules:
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                try:
                    module = _resolve_module(model, path)
                except AttributeError:
                    logger.warning(f"Cannot find module path: {path}")
                    continue

                prefix = path + "."
                sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                if sub_state_dict:
                    module.load_state_dict(sub_state_dict, strict=True)
                    if _is_rank_zero():
                        print(f"Parameters loaded to module '{path}'")
                else:
                    logger.warning(f"Parameters not found in checkpoint for module '{path}'")
        else:
            try:
                model.load_state_dict(checkpoint, strict=True)
                if _is_rank_zero():
                    print("Loaded <full_model> model parameters")
            except Exception as e:
                raise RuntimeError(f"Loading full model failed: {e}") from e

        return model

    @staticmethod
    def print_freeze_status(model):
        """
        Print the freezing status of each parameter in the model.
        """
        for name, param in model.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name:60s}  |  {status}")

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        Use Accelerator to prepare distributed training components.
        """
        return accelerator.prepare(*components)

    @staticmethod
    def euclidean_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.linalg.norm(predicted - ground_truth)

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """Safely reset a dataloader iterator."""
        epoch_counter += 1

        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_counter)

        return iter(dataloader), epoch_counter

    @staticmethod
    def compute_grad_angle_with_stats(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> Tuple[float, float]:
        """
        Compute cosine angles between two groups of gradient vectors.
        """
        angle_degs = []

        grads_action = grads_a[0]
        grads_action = grads_action[:32, :7]
        grads_vl = grads_v[0]
        grads_vl = grads_vl[:32, :7]
        for g_a, g_v in zip(grads_action, grads_vl):
            dot = torch.sum(g_a * g_v)
            norm_a_sq = torch.sum(g_a * g_a)
            norm_v_sq = torch.sum(g_v * g_v)

            norm_a = torch.sqrt(norm_a_sq + 1e-16)
            norm_v = torch.sqrt(norm_v_sq + 1e-16)

            cos_sim = (dot / (norm_a * norm_v)).clamp(-1.0, 1.0)
            angle_rad = torch.acos(cos_sim)
            angle_deg = angle_rad * (180.0 / torch.pi)

            angle_degs.append(angle_deg.item())

        angle_degs_tensor = torch.tensor(angle_degs)
        mean_angle_deg = torch.mean(angle_degs_tensor).item()
        angle_variance = torch.sqrt(torch.var(angle_degs_tensor)).item()
        return mean_angle_deg, angle_variance

    @staticmethod
    def pcgrad_project(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Apply PCGrad projection to grads_v when gradients conflict with grads_a.
        """
        dot, norm_a_sq = 0.0, 0.0
        for g_a, g_v in zip(grads_a, grads_v):
            dot += torch.sum(g_a * g_v)
            norm_a_sq += torch.sum(g_a * g_a)

        if dot < 0:
            coeff = dot / (norm_a_sq + 1e-6)
            grads_v = [g_v - coeff * g_a for g_a, g_v in zip(grads_a, grads_v)]

        return grads_v

    @staticmethod
    def eval_qwenpi(qwenpi, dataloader, num_batches=20):
        """
        Evaluate QwenQFormerDiT model, computing IoU and action distance.
        """
        iou_scores = []
        action_distances = []
        count = 0

        dataset_iter = iter(dataloader)
        while count < num_batches:
            try:
                batch_samples = next(dataset_iter)
                count += 1
            except StopIteration:
                break

            images = [example["image"] for example in batch_samples]
            instructions = [example["lang"] for example in batch_samples]
            actions = [example["action"] for example in batch_samples]
            solutions = [example["solution"] for example in batch_samples]

            predicted_solutions, normalized_actions = qwenpi.predict_action_withCoT(
                images=images, instructions=instructions, use_ddim=False, num_ddim_steps=20
            )

            parsed_solutions = []
            for solution in predicted_solutions:
                parsed_solution = TrainerUtils.extract_json_from_string(solution)
                parsed_solutions.append(parsed_solution)

            for pred_dict, gt_dict in zip(parsed_solutions, solutions):
                pred_pick_bbox = torch.tensor(pred_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_pick_bbox = torch.tensor(gt_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                pred_place_bbox = torch.tensor(pred_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_place_bbox = torch.tensor(gt_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)

                pick_iou = box_iou(pred_pick_bbox, gt_pick_bbox).item()
                place_iou = box_iou(pred_place_bbox, gt_place_bbox).item()

                iou_scores.append({"pick_iou": pick_iou, "place_iou": place_iou})

            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            action_distance = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_action_distance = action_distance / num_pots
            action_distances.append(average_action_distance)

        avg_action_distance = np.mean(action_distances)
        return {"iou_scores": iou_scores, "average_action_distance": avg_action_distance}

    @staticmethod
    def extract_json_from_string(input_string):
        """
        Extract a JSON object from a string and parse it into a dictionary.
        """
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON decode failed: {e}")
                return None

        print("No valid JSON part found")
        return None


def is_main_process():
    rank = int(os.environ.get("RANK", 0))
    return rank == 0
