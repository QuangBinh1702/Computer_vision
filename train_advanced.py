"""
Script train 2 model cải tiến trên server.
Chạy: python train_advanced.py

Tương đương notebook 03_advanced_training.ipynb nhưng chạy được trên terminal.
"""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from flowers102.data import build_dataloaders, build_transforms, build_mixup_cutmix_collate
from flowers102.models import create_model, freeze_feature_extractor, unfreeze_all
from flowers102.train import fit, TrainConfig, build_scheduler_with_warmup
from flowers102.evaluate import evaluate_model
from flowers102.utils import set_seed, save_json


# ============================================================
# CẤU HÌNH — chỉnh ở đây nếu cần
# ============================================================
DATA_DIR = ROOT / "flower_data" / "flower_data"
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 102
NUM_WORKERS = 4          # server thường có nhiều CPU cores

MODELS = [
    "convnext_tiny",     # Người 1
    "efficientnet_b3",   # Người 2
]

# Nếu chỉ muốn train 1 model, comment dòng còn lại:
# MODELS = ["convnext_tiny"]
# MODELS = ["efficientnet_b3"]


def train_model_2stage(model_name: str):
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Data ---
    train_tf, eval_tf = build_transforms(
        image_size=IMAGE_SIZE,
        use_augmentation=True,
        randaugment=True,
        randaugment_num_ops=2,
        randaugment_magnitude=9,
        random_erasing=0.1,
    )
    mixup_cutmix_collate = build_mixup_cutmix_collate(
        num_classes=NUM_CLASSES, mixup_alpha=0.2, cutmix_alpha=1.0,
    )
    train_loader, valid_loader, test_loader, train_ds, _, _ = build_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        use_augmentation=True,
        collate_fn=mixup_cutmix_collate,
    )
    print(f"Train: {len(train_ds)} images | Classes: {NUM_CLASSES}")

    # --- Model ---
    model = create_model(model_name, num_classes=NUM_CLASSES, pretrained=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- Stage 1: Freeze backbone, train head (5 epochs) ---
    print(f"\n--- Stage 1: Freeze backbone, train head (5 epochs) ---")
    freeze_feature_extractor(model)
    opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-4)
    sch1 = build_scheduler_with_warmup(opt1, total_epochs=5, warmup_epochs=1)
    cfg1 = TrainConfig(
        epochs=5, device=device,
        checkpoint_path=str(ROOT / "checkpoints" / "advanced" / f"{model_name}_stage1.pth"),
        use_amp=True, grad_clip_max_norm=1.0,
    )
    hist1 = fit(model, train_loader, valid_loader, criterion, opt1, sch1, cfg1)

    # --- Stage 2: Full fine-tune (20 epochs) ---
    print(f"\n--- Stage 2: Full fine-tune (20 epochs) ---")
    unfreeze_all(model)
    opt2 = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch2 = build_scheduler_with_warmup(opt2, total_epochs=20, warmup_epochs=3)
    cfg2 = TrainConfig(
        epochs=20, device=device,
        checkpoint_path=str(ROOT / "checkpoints" / "advanced" / f"{model_name}_best.pth"),
        early_stopping_patience=7, use_amp=True, grad_clip_max_norm=1.0,
    )
    hist2 = fit(model, train_loader, valid_loader, criterion, opt2, sch2, cfg2)

    # --- Test ---
    print(f"\n--- Test evaluation ---")
    ckpt = torch.load(
        ROOT / "checkpoints" / "advanced" / f"{model_name}_best.pth",
        map_location=device, weights_only=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, criterion, device=device)
    print(f">>> {model_name} | Test Top-1: {test_metrics['top1']:.4f} | Test Top-5: {test_metrics['top5']:.4f}")

    return {
        "model_name": model_name,
        "stage1_history": hist1,
        "stage2_history": hist2,
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "test_loss": test_metrics["loss"],
    }


def main():
    print("=" * 60)
    print("  Oxford Flowers 102 — Advanced Training")
    print("=" * 60)
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Models to train: {MODELS}")
    print()

    all_results = {}
    for model_name in MODELS:
        result = train_model_2stage(model_name)
        all_results[model_name] = result
        save_json(all_results, ROOT / "reports" / "advanced_all_results.json")

    # Final summary
    print("\n" + "=" * 60)
    print("  KẾT QUẢ CUỐI CÙNG")
    print("=" * 60)
    for name, res in all_results.items():
        print(f"  {name:20s}  Top-1: {res['test_top1']*100:.2f}%  Top-5: {res['test_top5']*100:.2f}%")
    print("=" * 60)
    print(f"Results saved to: {ROOT / 'reports' / 'advanced_all_results.json'}")


if __name__ == "__main__":
    main()
