from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import build_dataloaders
from .evaluate import evaluate_model
from .models import create_model
from .train import TrainConfig, fit
from .utils import save_json, set_seed


def run_baseline(data_dir: Path, out_dir: Path) -> None:
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, valid_loader, test_loader, train_ds, _, _ = build_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=0,
        image_size=224,
        use_augmentation=True,
    )

    model = create_model("efficientnet_b0", num_classes=len(train_ds.classes), pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=12)
    ckpt = out_dir / "checkpoints" / "baseline_best.pth"

    hist = fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=TrainConfig(
            epochs=12,
            device=device,
            checkpoint_path=str(ckpt),
            early_stopping_patience=4,
        ),
    )

    metrics = evaluate_model(model, test_loader, criterion=criterion, device=device)
    save_json(hist, out_dir / "reports" / "baseline_history.json")
    save_json(metrics, out_dir / "reports" / "baseline_test_metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Oxford Flowers 102 experiments")
    parser.add_argument("--data-dir", default="flower_data/flower_data")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--mode", choices=["baseline"], default="baseline")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "baseline":
        run_baseline(data_dir=data_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()

