import bisect
import torch
from torch.utils.data import Dataset
import os
import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torchvision.transforms as T
to_pil = T.ToPILImage()

from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES

def collate_fn(batch, img_keys, state_key, action_key, task_key, resize_size):
    examples = []
    for _, b in enumerate(batch):
        example = {"image": []}
        example["action"] = b[action_key].cpu().numpy()
        example["lang"] = b[task_key]

        for k in img_keys:
            img_primary = to_pil(b[k][0]).resize((resize_size, resize_size))
            example["image"].append(img_primary)
            
        for k in b.keys():
            if k == state_key:
                example["state"] = b[k][0:1].cpu().numpy()
        examples.append(example)
    return examples

class MixtureDataset(Dataset):
    def __init__(self, datasets):
        """
        datasets: List[Dataset]
        """
        self.datasets = datasets
        # prefix sum of lengths，用于快速定位 index 属于哪个 dataset
        self.cumulative_sizes = self._compute_cumulative_sizes()

    def _compute_cumulative_sizes(self):
        sizes = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            sizes.append(total)
        return sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # 找到 idx 属于哪个 dataset
        ds_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if ds_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[ds_idx - 1]
        return self.datasets[ds_idx][sample_idx]


def get_lerobot_v3_datasets(
    data_cfg: dict,
):
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    action_horizon = data_cfg.action_horizon
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]

    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        repo_id = os.path.join(data_root_dir, d_name)
        ds_meta = LeRobotDatasetMetadata(repo_id)

        observation_keys = []
        for k in ds_meta.features.keys():
            if "observation" in k:
                observation_keys.append(k)
        delta_timestamps = {
            # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
            "action": [t / ds_meta.fps for t in range(action_horizon)],
        }
        for k in observation_keys:
            delta_timestamps[k] = [t / ds_meta.fps for t in range(action_horizon+1)]
        dataset_mixture.append(
            LeRobotDataset(
                repo_id,
                delta_timestamps=delta_timestamps,
            )
        )
    #[print(ds.num_episodes, ds.num_frames, i) for i, ds in enumerate(dataset_mixture)]
    return MixtureDataset(dataset_mixture)
    