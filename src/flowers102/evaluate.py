from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks: Tuple[int, ...] = (1, 5)):
    with torch.no_grad():
        max_k = max(ks)
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        result = []
        batch_size = targets.size(0)
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            result.append((correct_k / batch_size).item())
        return result


def evaluate_model(
    model: nn.Module,
    loader,
    criterion: nn.Module | None = None,
    device: str = "cuda",
) -> Dict[str, float]:
    device = device if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    total = 0
    running_loss = 0.0
    top1_sum = 0.0
    top5_sum = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)

            if criterion is not None:
                loss = criterion(logits, labels)
                running_loss += loss.item() * labels.size(0)

            top1, top5 = topk_accuracy(logits, labels, ks=(1, 5))
            top1_sum += top1 * labels.size(0)
            top5_sum += top5 * labels.size(0)
            total += labels.size(0)

    return {
        "loss": running_loss / max(1, total) if criterion is not None else 0.0,
        "top1": top1_sum / max(1, total),
        "top5": top5_sum / max(1, total),
    }

