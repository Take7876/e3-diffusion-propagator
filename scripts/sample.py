from __future__ import annotations

import argparse

import torch

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

from diffusion_models.config import load_config
from diffusion_models.data.argon_dataset import ArgonTrajectoryDataset, collate_argon_samples
from diffusion_models.models import DiffusionPropagator
from diffusion_models.utils import describe_device, get_device

DATASET_CONFIG_KEYS = {"topology_path", "trajectory_path", "stride", "time_lag", "max_frames", "max_atoms", "box_wrap"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample one propagation step from a trained diffusion model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--frame-index", type=int, default=0, help="Dataset frame index to condition on.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cuda:0, mps, or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = ArgonTrajectoryDataset(**{k: v for k, v in config["data"].items() if k in DATASET_CONFIG_KEYS})
    batch = collate_argon_samples([dataset[args.frame_index]])
    device = get_device(args.device)
    print(f"using device: {describe_device(device)}")
    batch = {key: value.to(device) for key, value in batch.items()}

    model = DiffusionPropagator(
        cutoff=config["data"]["cutoff"],
        neighbor_k=config["data"]["neighbor_k"],
        **config["model"],
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    predicted_positions, predicted_velocities = model.sample_step(
        batch["positions"], batch["velocities"], box=batch["box"]
    )
    print("predicted_positions_shape:", tuple(predicted_positions.shape))
    print("predicted_velocities_shape:", tuple(predicted_velocities.shape))
    print("position_rmse_to_target:", torch.sqrt(((predicted_positions - batch["next_positions"]) ** 2).mean()).item())
    print("velocity_rmse_to_target:", torch.sqrt(((predicted_velocities - batch["next_velocities"]) ** 2).mean()).item())


if __name__ == "__main__":
    main()
