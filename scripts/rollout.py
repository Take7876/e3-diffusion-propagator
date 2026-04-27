from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

import numpy as np
import torch

from diffusion_models.config import load_config
from diffusion_models.data.argon_dataset import ArgonTrajectoryDataset
from diffusion_models.models import DiffusionPropagator
from diffusion_models.utils import describe_device, get_device

DATASET_CONFIG_KEYS = {"topology_path", "trajectory_path", "stride", "time_lag", "max_frames", "max_atoms", "box_wrap"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out a trained propagator and write predicted trajectory files.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--frame-index", type=int, default=0, help="Initial dataset frame index.")
    parser.add_argument("--steps", type=int, default=10, help="Number of propagation steps.")
    parser.add_argument("--output-prefix", type=str, default="outputs/argon_diffusion/rollout")
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


def write_xyz(path: Path, positions: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for frame_index, frame in enumerate(positions):
            handle.write(f"{frame.shape[0]}\n")
            handle.write(f"predicted frame {frame_index}\n")
            for atom_position in frame:
                handle.write(f"Ar {atom_position[0]:.8f} {atom_position[1]:.8f} {atom_position[2]:.8f}\n")


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be >= 1.")

    config = load_config(args.config)
    dataset = ArgonTrajectoryDataset(**{k: v for k, v in config["data"].items() if k in DATASET_CONFIG_KEYS})
    sample = dataset[args.frame_index]
    device = get_device(args.device)
    print(f"using device: {describe_device(device)}")
    model = load_model(config, args.checkpoint, device)

    positions = sample.positions.unsqueeze(0).to(device)
    velocities = sample.velocities.unsqueeze(0).to(device)
    box = sample.box.unsqueeze(0).to(device)

    predicted_positions = [positions.squeeze(0).cpu().numpy()]
    predicted_velocities = [velocities.squeeze(0).cpu().numpy()]

    with torch.no_grad():
        for _ in range(args.steps):
            positions, velocities = model.sample_step(positions, velocities, box=box)
            predicted_positions.append(positions.squeeze(0).cpu().numpy())
            predicted_velocities.append(velocities.squeeze(0).cpu().numpy())

    positions_array = np.stack(predicted_positions, axis=0)
    velocities_array = np.stack(predicted_velocities, axis=0)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    npz_path = output_prefix.with_suffix(".npz")
    xyz_path = output_prefix.with_suffix(".xyz")
    np.savez_compressed(
        npz_path,
        positions=positions_array,
        velocities=velocities_array,
        box=sample.box.numpy(),
        frame_index=args.frame_index,
        steps=args.steps,
        dt=sample.dt.item(),
    )
    write_xyz(xyz_path, positions_array)

    print("predicted_positions_shape:", positions_array.shape)
    print("predicted_velocities_shape:", velocities_array.shape)
    print("wrote:", npz_path)
    print("wrote:", xyz_path)


if __name__ == "__main__":
    main()
