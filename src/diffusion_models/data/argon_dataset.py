from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class ArgonSample:
    positions: torch.Tensor
    velocities: torch.Tensor
    next_positions: torch.Tensor
    next_velocities: torch.Tensor
    dt: torch.Tensor
    box: torch.Tensor


class ArgonTrajectoryDataset(Dataset[ArgonSample]):
    def __init__(
        self,
        topology_path: str,
        trajectory_path: str,
        stride: int = 1,
        time_lag: int = 1,
        max_frames: int | None = None,
        max_atoms: int | None = None,
        box_wrap: bool = False,
    ) -> None:
        import MDAnalysis as mda

        self.topology_path = Path(topology_path)
        self.trajectory_path = Path(trajectory_path)
        self.max_atoms = max_atoms
        self.box_wrap = box_wrap
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}.")
        if time_lag < 1:
            raise ValueError(f"time_lag must be >= 1, got {time_lag}.")
        self.time_lag = time_lag
        if not self.topology_path.exists():
            raise FileNotFoundError(f"Topology file not found: {self.topology_path}")
        if not self.trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {self.trajectory_path}")
        universe = mda.Universe(str(self.topology_path), str(self.trajectory_path))

        positions: list[np.ndarray] = []
        velocities: list[np.ndarray | None] = []
        boxes: list[np.ndarray] = []
        times: list[float] = []

        for frame_index, ts in enumerate(universe.trajectory[::stride]):
            if max_frames is not None and frame_index >= max_frames:
                break
            coords = ts.positions.astype(np.float32).copy()
            if box_wrap and hasattr(universe.atoms, "wrap"):
                universe.atoms.wrap(inplace=True)
                coords = universe.atoms.positions.astype(np.float32).copy()

            positions.append(coords)
            frame_velocities = getattr(ts, "velocities", None)
            velocities.append(None if frame_velocities is None else frame_velocities.astype(np.float32).copy())
            boxes.append(ts.dimensions[:3].astype(np.float32).copy())
            times.append(float(ts.time))

        if len(positions) <= time_lag:
            raise ValueError(
                f"At least time_lag + 1 frames are required; got {len(positions)} frames and time_lag={time_lag}."
            )

        self.positions = np.stack(positions[:-time_lag], axis=0)
        self.next_positions = np.stack(positions[time_lag:], axis=0)
        self.boxes = np.stack(boxes[:-time_lag], axis=0)
        self.next_boxes = np.stack(boxes[time_lag:], axis=0)
        times_array = np.asarray(times, dtype=np.float32)
        self.dt = times_array[time_lag:] - times_array[:-time_lag]

        if velocities[0] is None:
            all_positions = np.stack(positions, axis=0)
            frame_dt = np.diff(times_array)
            finite_difference = np.diff(all_positions, axis=0) / frame_dt[:, None, None]
            frame_velocities = np.concatenate([finite_difference, finite_difference[-1:]], axis=0)
            self.velocities = frame_velocities[:-time_lag]
            self.next_velocities = frame_velocities[time_lag:]
        else:
            if any(v is None for v in velocities):
                raise ValueError("Trajectory has velocities for only some frames.")
            vel_array = np.stack([v for v in velocities if v is not None], axis=0)
            self.velocities = vel_array[:-time_lag]
            self.next_velocities = vel_array[time_lag:]

        if self.max_atoms is not None:
            atom_count = self.positions.shape[1]
            if self.max_atoms > atom_count:
                raise ValueError(f"max_atoms={self.max_atoms} exceeds atom count {atom_count}.")

    def __len__(self) -> int:
        return len(self.positions)

    def _select_atoms(self, array: np.ndarray) -> np.ndarray:
        if self.max_atoms is None:
            return array
        return array[: self.max_atoms]

    def __getitem__(self, index: int) -> ArgonSample:
        positions = self._select_atoms(self.positions[index])
        velocities = self._select_atoms(self.velocities[index])
        next_positions = self._select_atoms(self.next_positions[index])
        next_velocities = self._select_atoms(self.next_velocities[index])

        center = positions.mean(axis=0, keepdims=True)
        next_center = next_positions.mean(axis=0, keepdims=True)

        return ArgonSample(
            positions=torch.from_numpy(positions - center),
            velocities=torch.from_numpy(velocities),
            next_positions=torch.from_numpy(next_positions - next_center),
            next_velocities=torch.from_numpy(next_velocities),
            dt=torch.tensor(self.dt[index], dtype=torch.float32),
            box=torch.from_numpy(self.boxes[index]),
        )


def collate_argon_samples(samples: list[ArgonSample]) -> dict[str, torch.Tensor]:
    return {
        "positions": torch.stack([sample.positions for sample in samples], dim=0),
        "velocities": torch.stack([sample.velocities for sample in samples], dim=0),
        "next_positions": torch.stack([sample.next_positions for sample in samples], dim=0),
        "next_velocities": torch.stack([sample.next_velocities for sample in samples], dim=0),
        "dt": torch.stack([sample.dt for sample in samples], dim=0),
        "box": torch.stack([sample.box for sample in samples], dim=0),
    }


def build_dataloaders(data_config: dict[str, Any], training_config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    dataset_kwargs = {
        key: value
        for key, value in data_config.items()
        if key in {"topology_path", "trajectory_path", "stride", "time_lag", "max_frames", "max_atoms", "box_wrap"}
    }
    dataset = ArgonTrajectoryDataset(**dataset_kwargs)
    if len(dataset) < 2:
        raise ValueError("The dataset must contain at least two transition samples for train/validation split.")
    train_size = max(1, int(len(dataset) * data_config.get("train_split", 0.9)))
    valid_size = len(dataset) - train_size
    if valid_size == 0:
        valid_size = 1
        train_size = len(dataset) - 1
    generator = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=generator)

    loader_kwargs = {
        "batch_size": training_config["batch_size"],
        "num_workers": training_config.get("num_workers", 0),
        "collate_fn": collate_argon_samples,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    return train_loader, valid_loader
