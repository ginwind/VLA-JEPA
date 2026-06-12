import json
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.training.trainer_utils import initialize_overwatch

overwatch = initialize_overwatch(__name__)


def auto_get_module_keys(module, max_depth=0, prefix_list=None, current_depth=0, current_prefix=""):
    """
    Get submodule keys from a module, with optional depth and prefix filtering.
    """
    if current_depth > max_depth:
        return []

    module_keys = []
    for name, sub_module in module.named_children():
        full_name = f"{current_prefix}.{name}" if current_prefix else name
        if prefix_list is None or any(full_name.startswith(prefix) for prefix in prefix_list):
            module_keys.append(full_name)
        module_keys.extend(auto_get_module_keys(sub_module, max_depth, prefix_list, current_depth + 1, full_name))
    return module_keys


def is_module_trainable(module):
    """
    Check whether a module's direct parameters are all trainable.
    """
    params = list(module.parameters(recurse=False))
    if params:
        return all(p.requires_grad for p in params)

    # Container modules with no direct parameters are treated as trainable;
    # submodule checks determine the final result.
    return True


def auto_get_trainable_modules(module, prefix="", max_depth=None):
    """
    Traverse the module and return trainable module names.

    If all submodules under a module are trainable, the parent module name is
    returned instead of every child name.
    """
    children = list(module.named_children())

    if (max_depth is not None and max_depth <= 0) or not children:
        return [prefix] if prefix and is_module_trainable(module) else []

    child_keys = []
    all_children_trainable = True
    for name, child in children:
        full_name = f"{prefix}.{name}" if prefix else name
        keys = auto_get_trainable_modules(child, full_name, None if max_depth is None else max_depth - 1)
        if not keys:
            if is_module_trainable(child):
                keys = [full_name]
            else:
                all_children_trainable = False
        elif len(keys) > 1:
            all_children_trainable = False
        child_keys.extend(keys)

    if is_module_trainable(module) and all_children_trainable and child_keys:
        return [prefix] if prefix else child_keys
    return child_keys


def print_freeze_status(self):
    """
    Print freeze status grouped by top-level module.
    """
    from collections import defaultdict

    status_dict = defaultdict(lambda: {"Frozen": 0, "Trainable": 0, "params": []})
    for full_name, param in self.named_parameters():
        top_module = full_name.split(".", 1)[0]
        state = "Frozen" if not param.requires_grad else "Trainable"
        status_dict[top_module]["params"].append((full_name, state))
        status_dict[top_module][state] += 1

    print("=== module parameter freezing status ===")
    for top_module, info in status_dict.items():
        frozen_count = info["Frozen"]
        trainable_count = info["Trainable"]

        if frozen_count > 0 and trainable_count == 0:
            print(f"{top_module:40s}  |  all Frozen ({frozen_count} parameters)")
        elif trainable_count > 0 and frozen_count == 0:
            print(f"{top_module:40s}  |  all Trainable ({trainable_count} parameters)")
        else:
            print(f"{top_module:40s}  |  mixed state -> Frozen: {frozen_count}, Trainable: {trainable_count}")
            for pname, pstate in info["params"]:
                print(f"    {pname:60s}  |  {pstate}")
    print("=========================\n")


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry = {}

    def register(self, key: str):
        """Decorator: register a builder function or class."""

        def decorator(framework_class):
            self._registry[key] = framework_class
            return framework_class

        return decorator

    def __contains__(self, key):
        return key in self._registry

    def __getitem__(self, key):
        return self._registry[key]

    def list(self):
        """List currently registered keys and objects."""
        return {k: v for k, v in self._registry.items()}


FRAMEWORK_REGISTRY = Registry("frameworks")


def _resolve_checkpoint_run_dir(pretrained_checkpoint):
    checkpoint_pt = Path(pretrained_checkpoint)
    if not checkpoint_pt.is_file():
        overwatch.error(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
        raise FileNotFoundError(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
    if checkpoint_pt.suffix != ".pt":
        raise ValueError(f"Expected a .pt checkpoint, got `{checkpoint_pt}`.")
    if len(checkpoint_pt.parents) < 2:
        raise ValueError(f"Checkpoint path `{checkpoint_pt}` must live under <run_dir>/checkpoints/.")

    overwatch.info(f"Loading from local checkpoint path `{checkpoint_pt}`")
    return checkpoint_pt.parents[1]


def _load_dataset_statistics(run_dir):
    dataset_statistics_json = run_dir / "dataset_statistics.json"
    if not dataset_statistics_json.exists():
        raise FileNotFoundError(f"Missing `dataset_statistics.json` for `{run_dir}`")

    with open(dataset_statistics_json, "r", encoding="utf-8") as f:
        return json.load(f)


def read_model_config(pretrained_checkpoint):
    """
    Load config.json and dataset statistics associated with a checkpoint.
    """
    run_dir = _resolve_checkpoint_run_dir(pretrained_checkpoint)
    config_json = run_dir / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(f"Missing `config.json` for `{run_dir}`")

    with open(config_json, "r", encoding="utf-8") as f:
        global_cfg = json.load(f)

    return global_cfg, _load_dataset_statistics(run_dir)


def read_mode_config(pretrained_checkpoint):
    """
    Load config.yaml and dataset statistics associated with a checkpoint.

    The misspelled function name is kept for backward compatibility with
    existing eval scripts.
    """
    run_dir = _resolve_checkpoint_run_dir(pretrained_checkpoint)
    config_yaml = run_dir / "config.yaml"
    if not config_yaml.exists():
        raise FileNotFoundError(f"Missing `config.yaml` for `{run_dir}`")

    try:
        ocfg = OmegaConf.load(str(config_yaml))
        global_cfg = OmegaConf.to_container(ocfg, resolve=True)
    except Exception as e:
        overwatch.error(f"Failed to load YAML config `{config_yaml}`: {e}")
        raise

    return global_cfg, _load_dataset_statistics(run_dir)
