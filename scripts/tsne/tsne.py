import argparse

import torch

import sys
import os
import wandb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from moment.common import PATHS
from moment.tasks.tsne import TSNE_
from moment.utils.config import Config
from moment.utils.utils import control_randomness, make_dir_if_not_exists, parse_config


def tsne(
    config_path: str = "configs/pretraining/tsne.yaml",
    default_config_path: str = "configs/default.yaml",
    gpu_id: int = 0,
) -> None:
    config = Config(
        config_file_path=config_path, default_config_file_path=default_config_path
    ).parse()

    control_randomness(config["random_seed"])

    config["device"] = gpu_id if torch.cuda.is_available() else "cpu"
    config["checkpoint_path"] = PATHS.CHECKPOINTS_DIR
    args = parse_config(config)
    make_dir_if_not_exists(config["checkpoint_path"])

    print(f"Running experiments with config:\n{args}\n")
    
    task_obj = TSNE_(args=args)

    NOTES = "tsne runs"
    task_obj.setup_logger(notes=NOTES) ## Wandb
    task_obj.train()
    task_obj.end_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretraining/tsne.yaml",
        help="Path to config file",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()
    wandb.init(mode="offline")
    tsne(config_path=args.config, gpu_id=args.gpu_id)
    
    # python scripts/tsne/tsne.py```