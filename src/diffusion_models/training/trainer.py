from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}


def evaluate(model: torch.nn.Module, loader, device: torch.device, use_amp: bool = False) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = model.loss(batch)
            total += float(loss.item())
            count += 1
    return total / max(count, 1)


def train(
    model: torch.nn.Module,
    train_loader,
    valid_loader,
    training_config: dict,
    output_dir: str,
    device: torch.device,
) -> None:
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
    )
    grad_clip_norm = training_config.get("grad_clip_norm")
    num_epochs = training_config["num_epochs"]
    log_every = training_config.get("log_every", 10)
    use_amp = bool(training_config.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_valid_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"epoch {epoch}")

        for step, batch in progress:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = model.loss(batch)
            scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

            if step % log_every == 0 or step == len(train_loader):
                progress.set_postfix(train_loss=running_loss / step)

        valid_loss = evaluate(model, valid_loader, device, use_amp=use_amp)
        checkpoint = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "valid_loss": valid_loss,
        }
        torch.save(checkpoint, output_path / "last.pt")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(checkpoint, output_path / "best.pt")

        print(f"epoch={epoch} train_loss={running_loss / max(len(train_loader), 1):.6f} valid_loss={valid_loss:.6f}")
