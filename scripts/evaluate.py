from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from diffusion_models.config import load_config
from diffusion_models.data.argon_dataset import ArgonTrajectoryDataset, collate_argon_samples
from diffusion_models.models import DiffusionPropagator
from diffusion_models.utils import describe_device, get_device

DATASET_CONFIG_KEYS = {"topology_path", "trajectory_path", "stride", "time_lag", "max_frames", "max_atoms", "box_wrap"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one-step propagation errors.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--output-csv", type=str, default="outputs/argon_diffusion/evaluation.csv")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cuda:0, mps, or cpu.")
    return parser.parse_args()


def load_model(config: dict, checkpoint_path: str, device: torch.device) -> DiffusionPropagator:
    model = DiffusionPropagator(
        cutoff=config["data"]["cutoff"],
        neighbor_k=config["data"]["neighbor_k"],
        **config["model"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = ArgonTrajectoryDataset(**{k: v for k, v in config["data"].items() if k in DATASET_CONFIG_KEYS})
    end_index = min(len(dataset), args.start_index + args.num_samples)
    indices = list(range(args.start_index, end_index))
    if not indices:
        raise ValueError("No samples selected for evaluation.")

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_argon_samples,
    )
    device = get_device(args.device)
    print(f"using device: {describe_device(device)}")
    model = load_model(config, args.checkpoint, device)

    rows: list[dict[str, float | int]] = []
    position_sse = 0.0
    velocity_sse = 0.0
    element_count = 0

    with torch.no_grad():
        for batch_offset, batch in enumerate(tqdm(loader, desc="evaluating")):
            batch = {key: value.to(device) for key, value in batch.items()}
            predicted_positions, predicted_velocities = model.sample_step(
                batch["positions"], batch["velocities"], box=batch["box"]
            )
            position_error = predicted_positions - batch["next_positions"]
            velocity_error = predicted_velocities - batch["next_velocities"]

            for item in range(position_error.shape[0]):
                global_index = indices[batch_offset * args.batch_size + item]
                pos_rmse = torch.sqrt(position_error[item].pow(2).mean()).item()
                vel_rmse = torch.sqrt(velocity_error[item].pow(2).mean()).item()
                rows.append(
                    {
                        "index": global_index,
                        "dt": batch["dt"][item].item(),
                        "position_rmse": pos_rmse,
                        "velocity_rmse": vel_rmse,
                    }
                )

            position_sse += position_error.pow(2).sum().item()
            velocity_sse += velocity_error.pow(2).sum().item()
            element_count += position_error.numel()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "dt", "position_rmse", "velocity_rmse"])
        writer.writeheader()
        writer.writerows(rows)

    print("samples:", len(rows))
    print("mean_position_rmse:", (position_sse / element_count) ** 0.5)
    print("mean_velocity_rmse:", (velocity_sse / element_count) ** 0.5)
    print("wrote:", output_path)


if __name__ == "__main__":
    main()
