from __future__ import annotations

import torch
from torch import nn

from diffusion_models.models.egnn import EGNNScoreNetwork


class DiffusionPropagator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        time_embedding_dim: int,
        cutoff: float,
        neighbor_k: int,
        diffusion_steps: int,
        beta_start: float,
        beta_end: float,
    ) -> None:
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.score_model = EGNNScoreNetwork(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_embedding_dim=time_embedding_dim,
            cutoff=cutoff,
            neighbor_k=neighbor_k,
        )

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def q_sample(
        self,
        clean_displacements: torch.Tensor,
        clean_delta_v: torch.Tensor,
        timesteps: torch.Tensor,
        noise_x: torch.Tensor,
        noise_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale_clean = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1)
        scale_noise = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1)
        noisy_displacements = scale_clean * clean_displacements + scale_noise * noise_x
        noisy_delta_v = scale_clean * clean_delta_v + scale_noise * noise_v
        return noisy_displacements, noisy_delta_v

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        positions = batch["positions"]
        velocities = batch["velocities"]
        box = batch.get("box")
        clean_displacements = batch["next_positions"] - positions
        clean_delta_v = batch["next_velocities"] - velocities

        batch_size = positions.shape[0]
        timesteps = torch.randint(0, self.diffusion_steps, (batch_size,), device=positions.device)
        noise_x = torch.randn_like(clean_displacements)
        noise_v = torch.randn_like(clean_delta_v)
        noisy_displacements, noisy_delta_v = self.q_sample(
            clean_displacements, clean_delta_v, timesteps, noise_x, noise_v
        )

        predicted_noise_x, predicted_noise_v = self.score_model(
            positions=positions,
            velocities=velocities,
            noisy_displacements=noisy_displacements,
            noisy_delta_v=noisy_delta_v,
            timesteps=timesteps,
            box=box,
        )

        loss_x = (predicted_noise_x - noise_x).pow(2).mean()
        loss_v = (predicted_noise_v - noise_v).pow(2).mean()
        return loss_x + loss_v

    @torch.no_grad()
    def sample_step(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        box: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        displacement = torch.randn_like(positions)
        delta_v = torch.randn_like(velocities)

        for step in reversed(range(self.diffusion_steps)):
            timesteps = torch.full((positions.shape[0],), step, device=positions.device, dtype=torch.long)
            predicted_noise_x, predicted_noise_v = self.score_model(
                positions=positions,
                velocities=velocities,
                noisy_displacements=displacement,
                noisy_delta_v=delta_v,
                timesteps=timesteps,
                box=box,
            )

            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]

            displacement = (displacement - beta * predicted_noise_x / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
            delta_v = (delta_v - beta * predicted_noise_v / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)

            if step > 0:
                displacement = displacement + torch.sqrt(beta) * torch.randn_like(displacement)
                delta_v = delta_v + torch.sqrt(beta) * torch.randn_like(delta_v)

        next_positions = positions + displacement
        next_positions = next_positions - next_positions.mean(dim=1, keepdim=True)
        next_velocities = velocities + delta_v
        return next_positions, next_velocities
