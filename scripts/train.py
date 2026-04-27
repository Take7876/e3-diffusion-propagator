from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_src_to_path

add_project_src_to_path()

from diffusion_models.config import load_config
from diffusion_models.data import build_dataloaders
from diffusion_models.models import DiffusionPropagator
from diffusion_models.training import train
from diffusion_models.utils import describe_device, get_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an E(3)-equivariant diffusion propagator.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cuda:0, mps, or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.get("seed", 42))

    data_config = dict(config["data"])
    train_split = data_config.pop("train_split", 0.9)
    loaders_config = dict(data_config)
    loaders_config["train_split"] = train_split

    train_loader, valid_loader = build_dataloaders(loaders_config, config["training"])
    model = DiffusionPropagator(
        cutoff=data_config["cutoff"],
        neighbor_k=data_config["neighbor_k"],
        **config["model"],
    )
    device = get_device(args.device)
    print(f"using device: {describe_device(device)}")

    output_dir = Path(config["output_dir"])
    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        training_config=config["training"],
        output_dir=str(output_dir),
        device=device,
    )


if __name__ == "__main__":
    main()
