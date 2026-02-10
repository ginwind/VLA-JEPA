import json
import os
from accelerate.logging import get_logger
from functools import partial
import torch
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            action_horizon=cfg.framework.action_model.action_horizon,
            video_horizon=cfg.framework.vj2_model.num_frames)
        
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=8,
            # shuffle=True
        )        
        if dist.get_rank() == 0: 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
    elif dataset_py == "lerobot_v3_datasets":
        from starVLA.dataloader.lerobot_v3_datasets import get_lerobot_v3_datasets, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_lerobot_v3_datasets(data_cfg=vla_dataset_cfg)

        custom_collate_fn = partial(collate_fn, 
            img_keys=cfg.datasets.vla_data.img_keys,
            state_key=cfg.datasets.vla_data.state_key if "state_key" in cfg.datasets.vla_data else None,
            action_key=cfg.datasets.vla_data.action_key if cfg.datasets.vla_data.action_key else None,
            task_key=cfg.datasets.vla_data.task_key if cfg.datasets.vla_data.task_key else None,
            resize_size=cfg.datasets.vla_data.resize_size)



        train_sampler = torch.utils.data.distributed.DistributedSampler(vla_dataset, shuffle=True)

        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=custom_collate_fn,
            num_workers=16,
            sampler=train_sampler,
        )      
        #if dist.get_rank() == 0: 
        #    for batch in vla_train_dataloader:
        #        print(batch)
        #        for k, v in batch.items():
        #            print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
        #        break
        return vla_train_dataloader
    elif dataset_py == "video_datasets":
        from starVLA.dataloader.video_datasets import VideoFolderDataset, collate_fn

        video_dataset_cfg = cfg.datasets.video_data

        video_dataset = VideoFolderDataset(
            video_dir=video_dataset_cfg.video_dir,
            text_file=video_dataset_cfg.text_file,
            n_frames=cfg.framework.vj2_model.num_frames,
            extensions=tuple(video_dataset_cfg.extensions),
            crop_h_size=video_dataset_cfg.video_resolution_size,
            crop_w_size=video_dataset_cfg.video_resolution_size,
            max_retry=10,
        )

        video_collate_fn = partial(collate_fn, 
            n_views=2,
            resolution_size=video_dataset_cfg.resolution_size)

        train_sampler = torch.utils.data.distributed.DistributedSampler(video_dataset, shuffle=True)

        video_train_dataloader = DataLoader(
            video_dataset,
            batch_size=video_dataset_cfg.per_device_batch_size,
            collate_fn=video_collate_fn,
            num_workers=16,
            sampler=train_sampler,
        )        
        return video_train_dataloader
