"""
📊 Đánh giá đầy đủ 3 model trên test set — Accuracy, F1, Precision, Recall, ECE.
Chạy: python evaluate_models.py

Kết quả lưu vào reports/evaluation/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from flowers102.data import build_dataloaders
from flowers102.evaluate import collect_predictions, compute_classification_metrics
from flowers102.models import create_model
from flowers102.utils import set_seed, save_json

# ── Config ───────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "flower_data" / "flower_data"
CHECKPOINTS = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "reports" / "evaluation"
NUM_CLASSES = 102

CAT_TO_NAME: dict[str, str] = json.loads(
    (ROOT / "cat_to_name.json").read_text(encoding="utf-8")
)

MODELS = [
    {
        "key": "convnext_tiny",
        "display_name": "ConvNeXt-Tiny",
        "arch": "convnext_tiny",
        "ckpt": CHECKPOINTS / "advanced" / "convnext_tiny_best.pth",
    },
    {
        "key": "efficientnet_b3",
        "display_name": "EfficientNet-B3",
        "arch": "efficientnet_b3",
        "ckpt": CHECKPOINTS / "advanced" / "efficientnet_b3_best.pth",
    },
    {
        "key": "efficientnet_b0",
        "display_name": "Baseline (B0)",
        "arch": "efficientnet_b0",
        "ckpt": CHECKPOINTS / "baseline_best.pth",
    },
]


def load_model(arch: str, ckpt_path: Path, device: str) -> nn.Module:
    model = create_model(arch, num_classes=NUM_CLASSES, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def plot_reliability_diagram(all_metrics: dict, save_path: Path) -> None:
    n = len(all_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, m) in zip(axes, all_metrics.items()):
        bins = m["ece_bins"]
        non_empty = [b for b in bins if b["count"] > 0]
        confs = [b["avg_confidence"] for b in non_empty]
        accs = [b["accuracy"] for b in non_empty]

        ax.bar(confs, accs, width=0.06, alpha=0.7, color="#4c72b0", label="Accuracy")
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{name}\nECE = {m['ece']:.4f}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Reliability Diagrams", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Reliability diagram → {save_path}")


def plot_comparison_bar(comparison_df: pd.DataFrame, save_path: Path) -> None:
    metrics = ["Accuracy (%)", "Macro F1", "Weighted F1"]
    models = comparison_df.index.tolist()
    x = np.arange(len(metrics))
    width = 0.22
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [comparison_df.loc[model, m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width=width * 0.9, label=model,
                      color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Accuracy & F1", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Comparison chart → {save_path}")


def plot_confusion_matrix(cm: list, class_names: list, model_name: str, save_path: Path) -> None:
    cm_arr = np.array(cm)
    row_sums = cm_arr.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm_arr / row_sums

    # Show top 20 most-confused classes
    diag = np.diag(cm_arr)
    error_counts = cm_arr.sum(axis=1) - diag
    top_idx = np.argsort(error_counts)[::-1][:20]
    top_idx = np.sort(top_idx)

    cm_sub = cm_norm[np.ix_(top_idx, top_idx)]
    labels = [class_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_sub, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels,
                cmap="Blues", linewidths=0.3, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name} (Top 20 most-confused)", fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Confusion matrix → {save_path}")


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build test loader
    _, _, test_loader, _, _, test_ds = build_dataloaders(
        data_dir=DATA_DIR, batch_size=32, num_workers=0,
        image_size=224, use_augmentation=False,
    )

    # Derive class names from dataset (consistent ordering)
    class_names = [CAT_TO_NAME[folder] for folder in test_ds.classes]

    print(f"Test set: {len(test_ds)} images, {NUM_CLASSES} classes\n")

    all_metrics = {}
    summary = {}

    for cfg in MODELS:
        print(f"{'='*50}")
        print(f"  Evaluating: {cfg['display_name']}")
        print(f"{'='*50}")

        model = load_model(cfg["arch"], cfg["ckpt"], device)
        payload = collect_predictions(model, test_loader, device=device)

        metrics = compute_classification_metrics(
            probs=payload["probs"],
            preds=payload["preds"],
            targets=payload["targets"],
            class_names=class_names,
        )

        # Print summary
        print(f"  Top-1 Accuracy:     {metrics['accuracy']*100:.2f}%")
        print(f"  Top-5 Accuracy:     {metrics['top5_accuracy']*100:.2f}%")
        print(f"  Macro F1:           {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:        {metrics['weighted_f1']:.4f}")
        print(f"  Macro Precision:    {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:       {metrics['macro_recall']:.4f}")
        print(f"  ECE:                {metrics['ece']:.4f}")
        print()

        all_metrics[cfg["display_name"]] = metrics

        # Save per-model detailed results
        model_dir = OUTPUT_DIR / cfg["key"]
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save full metrics (excluding large arrays for JSON)
        save_json({
            "model": cfg["display_name"],
            "accuracy": metrics["accuracy"],
            "top5_accuracy": metrics["top5_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "ece": metrics["ece"],
            "ece_bins": metrics["ece_bins"],
        }, model_dir / "summary.json")

        # Per-class metrics CSV
        report = metrics["classification_report"]
        per_class_rows = []
        for idx, name in enumerate(class_names):
            if name in report:
                r = report[name]
                per_class_rows.append({
                    "class_idx": idx,
                    "class_name": name,
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1-score": r["f1-score"],
                    "support": int(r["support"]),
                })
        pd.DataFrame(per_class_rows).to_csv(model_dir / "per_class_metrics.csv", index=False)

        # Confusion matrix
        plot_confusion_matrix(
            metrics["confusion_matrix"], class_names,
            cfg["display_name"], model_dir / "confusion_matrix.png",
        )

        # Compact summary for demo app
        summary[cfg["key"]] = {
            "display_name": cfg["display_name"],
            "accuracy": metrics["accuracy"],
            "top5_accuracy": metrics["top5_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "ece": metrics["ece"],
        }

    # Save model_comparison.json for demo_app.py
    save_json(summary, OUTPUT_DIR / "model_comparison.json")
    print(f"\n[✓] Summary JSON → {OUTPUT_DIR / 'model_comparison.json'}")

    # Comparison table
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Model": name,
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "Top-5 (%)": round(m["top5_accuracy"] * 100, 2),
            "Macro F1": round(m["macro_f1"], 4),
            "Weighted F1": round(m["weighted_f1"], 4),
            "Macro Precision": round(m["macro_precision"], 4),
            "Macro Recall": round(m["macro_recall"], 4),
            "ECE ↓": round(m["ece"], 4),
        })
    comparison_df = pd.DataFrame(rows).set_index("Model")
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv")
    print(f"[✓] Comparison CSV → {OUTPUT_DIR / 'model_comparison.csv'}")

    print(f"\n{'='*60}")
    print("  KẾT QUẢ SO SÁNH")
    print(f"{'='*60}")
    print(comparison_df.to_string())
    print(f"{'='*60}")

    # Plots
    plot_comparison_bar(comparison_df, OUTPUT_DIR / "comparison_metrics.png")
    plot_reliability_diagram(all_metrics, OUTPUT_DIR / "reliability_diagram.png")

    print("\n✅ Đánh giá hoàn tất! Kết quả lưu tại:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
