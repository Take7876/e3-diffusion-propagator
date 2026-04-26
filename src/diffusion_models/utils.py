from __future__ import annotations

import math
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        memory_gb = torch.cuda.get_device_properties(index).total_memory / 1024**3
        return f"cuda:{index} ({name}, {memory_gb:.1f} GB)"
    return str(device)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    exponent = -math.log(10000.0) / max(half_dim - 1, 1)
    frequencies = torch.exp(
        torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent
    )
    angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding
