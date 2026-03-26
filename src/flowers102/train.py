from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .evaluate import evaluate_model


@dataclass
class TrainConfig:
    epochs: int = 10
    device: str = "cuda"
    checkpoint_path: str = "checkpoints/best.pth"
    early_stopping_patience: Optional[int] = None
    use_amp: bool = False
    grad_clip_max_norm: float = 0.0


def _train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_clip_max_norm: float = 0.0,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    use_amp = scaler is not None

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        if labels.ndim == 1:
            running_correct += (preds == labels).sum().item()
        else:
            running_correct += (preds == labels.argmax(dim=1)).sum().item()
        total += batch_size

    return {
        "loss": running_loss / max(1, total),
        "top1": running_correct / max(1, total),
    }


def fit(
    model: nn.Module,
    train_loader,
    valid_loader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    config: TrainConfig,
) -> Dict[str, List[float]]:
    device = config.device if torch.cuda.is_available() else "cpu"
    model.to(device)

    history = {
        "train_loss": [],
        "train_top1": [],
        "valid_loss": [],
        "valid_top1": [],
        "valid_top5": [],
    }

    best_valid_top1 = -1.0
    wait = 0
    ckpt_path = Path(config.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    scaler = (
        torch.amp.GradScaler("cuda") if config.use_amp and device == "cuda" else None
    )

    for epoch in range(config.epochs):
        train_stats = _train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler,
            grad_clip_max_norm=config.grad_clip_max_norm,
        )
        valid_stats = evaluate_model(model, valid_loader, criterion, device=device)

        history["train_loss"].append(train_stats["loss"])
        history["train_top1"].append(train_stats["top1"])
        history["valid_loss"].append(valid_stats["loss"])
        history["valid_top1"].append(valid_stats["top1"])
        history["valid_top5"].append(valid_stats["top5"])

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch + 1}/{config.epochs}]  "
            f"train_loss={train_stats['loss']:.4f}  train_top1={train_stats['top1']:.4f}  "
            f"valid_loss={valid_stats['loss']:.4f}  valid_top1={valid_stats['top1']:.4f}  "
            f"valid_top5={valid_stats['top5']:.4f}  lr={current_lr:.6f}"
        )

        if valid_stats["top1"] > best_valid_top1:
            best_valid_top1 = valid_stats["top1"]
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_valid_top1": best_valid_top1,
                    "history": history,
                },
                ckpt_path,
            )
        else:
            wait += 1

        if config.early_stopping_patience is not None and wait >= config.early_stopping_patience:
            break

    return history


def build_scheduler_with_warmup(
    optimizer: Optimizer,
    total_epochs: int,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
) -> _LRScheduler:
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

