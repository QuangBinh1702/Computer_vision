"""
🌸 Flower Recognition Demo — Oxford Flowers 102
Streamlit app để demo nhận diện hoa với ConvNeXt-Tiny & EfficientNet-B3.
Chạy: streamlit run demo_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.flowers102.models import create_model

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "flower_data" / "flower_data"
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "checkpoints"

# ── Class-name mapping ──────────────────────────────────────────────────
# cat_to_name.json maps folder id (str) → flower name
# ImageFolder sorts folder names lexicographically: "1","10","100",…
# We build idx→name using the same order.
CAT_TO_NAME: dict[str, str] = json.loads(
    (DATA_DIR / "cat_to_name.json").read_text(encoding="utf-8")
)
FOLDER_NAMES_SORTED = sorted(
    [p.name for p in (DATA_DIR / "train").iterdir() if p.is_dir()],
    key=str,                       # lexicographic — same as ImageFolder
)
IDX_TO_NAME = {
    idx: CAT_TO_NAME.get(folder, folder)
    for idx, folder in enumerate(FOLDER_NAMES_SORTED)
}

# ── Available models ────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "ConvNeXt-Tiny  (Top-1: 99.51%)": {
        "arch": "convnext_tiny",
        "ckpt": CHECKPOINTS / "advanced" / "convnext_tiny_best.pth",
    },
    "EfficientNet-B3  (Top-1: 98.78%)": {
        "arch": "efficientnet_b3",
        "ckpt": CHECKPOINTS / "advanced" / "efficientnet_b3_best.pth",
    },
    "EfficientNet-B0 Baseline": {
        "arch": "efficientnet_b0",
        "ckpt": CHECKPOINTS / "baseline_best.pth",
    },
}

# ── Transform (eval) ───────────────────────────────────────────────────
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


# ── Model loading (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_model(arch: str, ckpt_path: str) -> torch.nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(arch, num_classes=102, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def predict(model: torch.nn.Module, image: Image.Image, top_k: int = 5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    top = torch.topk(probs, top_k)
    results = []
    for prob, idx in zip(top.values, top.indices):
        results.append((IDX_TO_NAME[idx.item()], float(prob)))
    return results


# ═══════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🌸 Flower Recognition",
    page_icon="🌸",
    layout="wide",
)

st.title("🌸 Nhận diện hoa — Oxford Flowers 102")
st.markdown(
    "Upload ảnh hoa bất kỳ, model sẽ dự đoán loại hoa với **Top-5 confidence**."
)

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")
    model_choice = st.selectbox("Chọn model", list(MODEL_CONFIGS.keys()))
    top_k = st.slider("Số lượng Top-K", min_value=1, max_value=10, value=5)

    st.divider()
    st.header("📊 Kết quả so sánh")
    comparison = REPORTS / "model_comparison.png"
    if comparison.exists():
        st.image(str(comparison), use_container_width=True)

    st.divider()
    st.header("🛠️ Kỹ thuật cải tiến")
    techniques = [
        "2-Stage Fine-tuning (freeze → unfreeze)",
        "RandAugment (num_ops=2, magnitude=9)",
        "Mixup (α=0.2) + CutMix (α=1.0)",
        "Label Smoothing (ε=0.1)",
        "Cosine LR + Linear Warmup (3 epochs)",
        "AMP (Mixed Precision)",
        "Gradient Clipping (max_norm=1.0)",
        "Random Erasing (p=0.1)",
    ]
    for t in techniques:
        st.markdown(f"- {t}")

# ── Load model ──────────────────────────────────────────────────────────
cfg = MODEL_CONFIGS[model_choice]
model = load_model(cfg["arch"], str(cfg["ckpt"]))

# ── Main area ───────────────────────────────────────────────────────────
tab_predict, tab_compare, tab_gallery, tab_about = st.tabs(
    ["🔍 Dự đoán", "⚔️ So sánh Model", "🖼️ Gallery", "ℹ️ Giới thiệu"]
)

# ── Tab: Predict ────────────────────────────────────────────────────────
with tab_predict:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        uploaded = st.file_uploader(
            "Tải ảnh hoa lên", type=["jpg", "jpeg", "png", "webp"]
        )
        # Sample images from test set
        use_sample = st.checkbox("Hoặc dùng ảnh mẫu từ test set")
        sample_img = None
        if use_sample:
            test_dir = DATA_DIR / "test"
            sample_classes = sorted(test_dir.iterdir())[:10]
            sample_options = {}
            for cls_dir in sample_classes:
                imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
                if imgs:
                    name = CAT_TO_NAME.get(cls_dir.name, cls_dir.name)
                    sample_options[f"{name} ({cls_dir.name})"] = imgs[0]
            choice = st.selectbox("Chọn ảnh mẫu", list(sample_options.keys()))
            sample_img = sample_options[choice]

    # Determine which image to use
    image = None
    if uploaded is not None:
        image = Image.open(uploaded)
    elif use_sample and sample_img is not None:
        image = Image.open(sample_img)

    with col_result:
        if image is not None:
            st.image(image, caption="Ảnh đầu vào", use_container_width=True)

            with st.spinner("Đang dự đoán..."):
                results = predict(model, image, top_k=top_k)

            st.subheader("📋 Kết quả dự đoán")
            # Top-1
            top_name, top_conf = results[0]
            st.success(f"🌸 **{top_name}** — {top_conf:.1%}")

            # Progress bars for each prediction
            for rank, (name, conf) in enumerate(results, 1):
                col_name, col_bar = st.columns([1, 2])
                with col_name:
                    st.write(f"**#{rank}** {name}")
                with col_bar:
                    st.progress(conf, text=f"{conf:.2%}")

            st.caption(
                "ℹ️ Confidence 60–90% là bình thường — model được train với "
                "**Label Smoothing** và **Mixup/CutMix** để tránh overconfident, "
                "giúp tổng quát hóa tốt hơn trên ảnh mới. "
                "Accuracy trên tập test vẫn đạt **99.51%** (ConvNeXt) / **98.78%** (EfficientNet-B3)."
            )
        else:
            st.info("👈 Upload ảnh hoặc chọn ảnh mẫu để bắt đầu.")

# ── Tab: Compare ────────────────────────────────────────────────────────
with tab_compare:
    st.subheader("So sánh 3 model trên cùng 1 ảnh")
    st.markdown(
        "Upload 1 ảnh để xem cả 3 model dự đoán thế nào. "
        "**Baseline** confidence cao (99%) vì không dùng Label Smoothing/Mixup — "
        "nhưng accuracy tổng thể trên test set **thấp hơn** model cải tiến."
    )

    cmp_uploaded = st.file_uploader(
        "Tải ảnh hoa lên", type=["jpg", "jpeg", "png", "webp"], key="cmp_upload"
    )
    cmp_use_sample = st.checkbox("Hoặc dùng ảnh mẫu", key="cmp_sample")
    cmp_image = None

    if cmp_use_sample:
        test_dir = DATA_DIR / "test"
        cmp_classes = sorted(test_dir.iterdir())[:15]
        cmp_options = {}
        for cls_dir in cmp_classes:
            imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
            if imgs:
                name = CAT_TO_NAME.get(cls_dir.name, cls_dir.name)
                cmp_options[f"{name} ({cls_dir.name})"] = imgs[0]
        cmp_choice = st.selectbox("Chọn ảnh mẫu", list(cmp_options.keys()), key="cmp_sel")
        cmp_image = Image.open(cmp_options[cmp_choice])
    if cmp_uploaded is not None:
        cmp_image = Image.open(cmp_uploaded)

    if cmp_image is not None:
        st.image(cmp_image, caption="Ảnh đầu vào", width=300)
        st.divider()

        cols = st.columns(3)
        compare_models = [
            ("ConvNeXt-Tiny", "convnext_tiny", CHECKPOINTS / "advanced" / "convnext_tiny_best.pth",
             "✅ Label Smoothing, Mixup, CutMix\n\nTest accuracy: **99.51%**"),
            ("EfficientNet-B3", "efficientnet_b3", CHECKPOINTS / "advanced" / "efficientnet_b3_best.pth",
             "✅ Label Smoothing, Mixup, CutMix\n\nTest accuracy: **98.78%**"),
            ("Baseline (B0)", "efficientnet_b0", CHECKPOINTS / "baseline_best.pth",
             "❌ Không có Label Smoothing, Mixup\n\nConfidence cao nhưng **overconfident**"),
        ]

        for col, (display_name, arch, ckpt_path, note) in zip(cols, compare_models):
            with col:
                st.markdown(f"### {display_name}")
                m = load_model(arch, str(ckpt_path))
                res = predict(m, cmp_image, top_k=3)
                top_name, top_conf = res[0]
                st.metric(label="Dự đoán", value=top_name, delta=f"{top_conf:.1%}")
                for rank, (name, conf) in enumerate(res, 1):
                    st.progress(conf, text=f"#{rank} {name} — {conf:.1%}")
                st.caption(note)

        st.info(
            "💡 **Baseline confidence 99%** không có nghĩa là model tốt hơn. "
            "Model cải tiến được train với **Label Smoothing + Mixup/CutMix** "
            "để phân bổ xác suất đều hơn (well-calibrated), giúp model **ít sai trên ảnh mới** hơn. "
            "Trên toàn bộ tập test, ConvNeXt (99.51%) vẫn chính xác hơn Baseline."
        )

# ── Tab: Gallery ────────────────────────────────────────────────────────
with tab_gallery:
    st.subheader("Các biểu đồ từ quá trình huấn luyện")
    gallery_images = {
        "Ảnh mẫu các loại hoa": "sample_flowers.png",
        "Data Augmentation demo": "augmentation_demo.png",
        "Training Curves": "training_curves.png",
        "So sánh Model": "model_comparison.png",
        "Phân phối lớp": "class_distribution.png",
        "t-SNE Visualization": "viz_tsne.png",
        "Radar Chart": "viz_radar.png",
        "Heatmap": "viz_heatmap.png",
        "Training 3D": "viz_training_3d.png",
    }

    cols = st.columns(3)
    for i, (caption, filename) in enumerate(gallery_images.items()):
        path = REPORTS / filename
        if path.exists():
            with cols[i % 3]:
                st.image(str(path), caption=caption, use_container_width=True)

# ── Tab: About ──────────────────────────────────────────────────────────
with tab_about:
    st.subheader("Giới thiệu dự án")
    st.markdown(
        """
        **Dataset**: Oxford Flowers 102 — 102 loại hoa, 8189 ảnh

        **Pipeline**:
        1. `01_data_audit` — Khảo sát & phân tích dữ liệu
        2. `02_baseline_train` — Huấn luyện baseline (EfficientNet-B0)
        3. `03_advanced_training` — Cải tiến với ConvNeXt-Tiny & EfficientNet-B3
        4. `04_hparam_search` — Tìm siêu tham số tối ưu
        5. `05_evaluation_error_analysis` — Đánh giá & phân tích lỗi
        6. `06_inference_demo` — Demo dự đoán

        **Kết quả**:

        | Model | Top-1 | Top-5 |
        |---|---|---|
        | EfficientNet-B3 | 98.78% | 100.00% |
        | **ConvNeXt-Tiny** | **99.51%** | **100.00%** |
        """
    )
    st.markdown("---")
    st.caption("Nguyễn Quang Bình — 102220095")
