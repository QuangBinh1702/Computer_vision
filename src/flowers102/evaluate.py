from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)


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


# ---------------------------------------------------------------------------
# Collect raw predictions (logits, probs, preds, targets) in a single pass
# ---------------------------------------------------------------------------

def collect_predictions(
    model: nn.Module,
    loader,
    criterion: nn.Module | None = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    device = device if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    total = 0
    running_loss = 0.0
    logits_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)

            if criterion is not None:
                loss = criterion(logits, labels)
                running_loss += loss.item() * labels.size(0)

            logits_all.append(logits.cpu())
            targets_all.append(labels.cpu())
            total += labels.size(0)

    logits_t = torch.cat(logits_all)
    targets_t = torch.cat(targets_all)
    probs_t = torch.softmax(logits_t, dim=1)
    preds_t = probs_t.argmax(dim=1)

    top1, top5 = topk_accuracy(logits_t, targets_t, ks=(1, 5))

    return {
        "loss": running_loss / max(1, total) if criterion is not None else 0.0,
        "logits": logits_t.numpy(),
        "probs": probs_t.numpy(),
        "preds": preds_t.numpy(),
        "targets": targets_t.numpy(),
        "top1": top1,
        "top5": top5,
    }


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def compute_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, Any]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == targets).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []

    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= left) & (confidences < right)
        else:
            mask = (confidences >= left) & (confidences <= right)

        count = int(mask.sum())
        if count == 0:
            bins.append({"bin": i, "count": 0, "avg_confidence": 0.0, "accuracy": 0.0})
            continue

        bin_conf = float(confidences[mask].mean())
        bin_acc = float(correctness[mask].mean())
        ece += abs(bin_acc - bin_conf) * (count / len(targets))

        bins.append({"bin": i, "count": count, "avg_confidence": bin_conf, "accuracy": bin_acc})

    return {"ece": float(ece), "bins": bins}


# ---------------------------------------------------------------------------
# Full classification metrics (Accuracy, F1, Precision, Recall, CM, ECE)
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    probs: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    labels = list(range(len(class_names)))

    report = classification_report(
        targets, preds,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(targets, preds, labels=labels)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        targets, preds, average="weighted", zero_division=0,
    )

    top5 = top_k_accuracy_score(targets, probs, k=5, labels=labels)
    ece_result = compute_ece(probs, targets)

    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "top5_accuracy": float(top5),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "ece": ece_result["ece"],
        "ece_bins": ece_result["bins"],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Original evaluate_model — kept for backward compatibility with train.py
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    loader,
    criterion: nn.Module | None = None,
    device: str = "cuda",
) -> Dict[str, float]:
    payload = collect_predictions(model, loader, criterion, device)
    return {
        "loss": payload["loss"],
        "top1": payload["top1"],
        "top5": payload["top5"],
    }

