from __future__ import annotations

import torch
from torch import nn

from diffusion_models.utils import sinusoidal_embedding


def build_neighbor_graph(
    positions: torch.Tensor,
    cutoff: float,
    neighbor_k: int,
    box: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_atoms, _ = positions.shape
    relative_all = positions.unsqueeze(1) - positions.unsqueeze(2)
    if box is not None:
        box = box.to(device=positions.device, dtype=positions.dtype)
        if box.ndim == 1:
            box = box.view(1, 1, 1, 3)
        elif box.ndim == 2:
            box = box.view(batch_size, 1, 1, 3)
        positive_box = box > 0
        safe_box = torch.where(positive_box, box, torch.ones_like(box))
        image_shift = torch.where(
            positive_box,
            safe_box * torch.round(relative_all / safe_box),
            torch.zeros_like(relative_all),
        )
        relative_all = relative_all - image_shift

    distances = relative_all.norm(dim=-1)
    eye = torch.eye(num_atoms, device=positions.device, dtype=torch.bool).unsqueeze(0)
    valid = (distances < cutoff) & (~eye)
    masked_distances = distances.masked_fill(~valid, float("inf"))
    knn_distances, knn_indices = torch.topk(masked_distances, k=min(neighbor_k, num_atoms - 1), largest=False)
    edge_mask = torch.isfinite(knn_distances)

    batch_index = torch.arange(batch_size, device=positions.device).view(batch_size, 1, 1)
    src_index = torch.arange(num_atoms, device=positions.device).view(1, num_atoms, 1).expand_as(knn_indices)
    relative = relative_all[batch_index, src_index, knn_indices]
    return knn_indices, relative, edge_mask


class EquivariantLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        neighbor_indices: torch.Tensor,
        relative: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_atoms, _, _ = relative.shape
        hidden_dim = node_features.shape[-1]
        gather_index = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
        neighbor_features = torch.gather(
            node_features.unsqueeze(1).expand(-1, num_atoms, -1, -1),
            2,
            gather_index,
        )
        src_features = node_features.unsqueeze(2).expand_as(neighbor_features)
        radial = (relative**2).sum(dim=-1, keepdim=True)
        edge_inputs = torch.cat([src_features, neighbor_features, radial], dim=-1)
        messages = self.edge_mlp(edge_inputs) * edge_mask.unsqueeze(-1)

        coord_scale = self.coord_mlp(messages)
        coord_updates = (relative * coord_scale * edge_mask.unsqueeze(-1)).sum(dim=2)
        positions = positions + coord_updates

        aggregated = messages.sum(dim=2)
        node_inputs = torch.cat([node_features, aggregated], dim=-1)
        node_features = node_features + self.node_mlp(node_inputs)
        return node_features, positions


class EGNNScoreNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        time_embedding_dim: int,
        cutoff: float,
        neighbor_k: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim
        self.cutoff = cutoff
        self.neighbor_k = neighbor_k
        self.input_proj = nn.Linear(6 + time_embedding_dim, hidden_dim)
        self.layers = nn.ModuleList([EquivariantLayer(hidden_dim) for _ in range(num_layers)])
        self.vector_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        noisy_displacements: torch.Tensor,
        noisy_delta_v: torch.Tensor,
        timesteps: torch.Tensor,
        box: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_atoms, _ = positions.shape
        time_features = sinusoidal_embedding(timesteps, self.time_embedding_dim)
        time_features = time_features.unsqueeze(1).expand(batch_size, num_atoms, -1)

        node_features = torch.cat(
            [
                velocities.norm(dim=-1, keepdim=True),
                noisy_displacements.norm(dim=-1, keepdim=True),
                noisy_delta_v.norm(dim=-1, keepdim=True),
                (velocities * noisy_displacements).sum(dim=-1, keepdim=True),
                (velocities * noisy_delta_v).sum(dim=-1, keepdim=True),
                (noisy_displacements * noisy_delta_v).sum(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        node_features = self.input_proj(torch.cat([node_features, time_features], dim=-1))

        current_positions = positions
        neighbor_indices, relative, edge_mask = build_neighbor_graph(
            current_positions, cutoff=self.cutoff, neighbor_k=self.neighbor_k, box=box
        )

        for layer in self.layers:
            node_features, current_positions = layer(
                node_features, current_positions, neighbor_indices, relative, edge_mask
            )
            neighbor_indices, relative, edge_mask = build_neighbor_graph(
                current_positions, cutoff=self.cutoff, neighbor_k=self.neighbor_k, box=box
            )

        vector_coeffs = self.vector_head(node_features)
        coeffs_x, coeffs_v = vector_coeffs[..., :3], vector_coeffs[..., 3:]
        basis = torch.stack([noisy_displacements, noisy_delta_v, velocities], dim=-2)

        predicted_eps_x = (coeffs_x.unsqueeze(-1) * basis).sum(dim=-2)
        predicted_eps_v = (coeffs_v.unsqueeze(-1) * basis).sum(dim=-2)
        return predicted_eps_x, predicted_eps_v
