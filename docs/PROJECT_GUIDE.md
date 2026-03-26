# 📖 Hướng dẫn chi tiết dự án Oxford Flowers 102

> Tài liệu giải thích **từng file** trong dự án: nó làm gì, tại sao tồn tại, chạy ra kết quả gì, và vì sao lại thiết kế như vậy.

---

## 📁 Tổng quan cấu trúc thư mục

```
CV/
├── configs/                    # Cấu hình huấn luyện (YAML)
│   ├── baseline.yaml
│   ├── advanced.yaml
│   └── search_space.yaml
├── docs/                       # Tài liệu
│   └── PROJECT_GUIDE.md        # (file này)
├── flower_data/flower_data/    # Dataset Oxford Flowers 102
│   ├── train/                  # 102 thư mục con (1 thư mục = 1 loại hoa)
│   ├── valid/
│   ├── test/
│   └── cat_to_name.json        # Mapping: số → tên hoa ("1" → "pink primrose")
├── notebooks/                  # 6 notebook chạy theo thứ tự
│   ├── 01_data_audit.ipynb
│   ├── 02_baseline_train.ipynb
│   ├── 03_advanced_training.ipynb
│   ├── 04_hparam_search.ipynb
│   ├── 05_evaluation_error_analysis.ipynb
│   └── 06_inference_demo.ipynb
├── src/flowers102/             # Source code chính (Python modules)
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── experiment.py
│   └── utils.py
├── checkpoints/                # Model weights (.pth) - tự tạo khi train
├── reports/                    # Kết quả thí nghiệm (JSON, PNG)
└── requirements.txt            # Thư viện Python cần cài
```

---

## 🧠 Tại sao lại tổ chức như vậy?

### Vì sao tách `src/` và `notebooks/`?

| Cách làm | Ưu điểm | Nhược điểm |
|---|---|---|
| ❌ Viết tất cả code trong 1 notebook duy nhất | Nhanh, đơn giản | Code lặp lại, khó bảo trì, notebook rất dài |
| ❌ Chỉ dùng file `.py` | Tái sử dụng tốt | Không trực quan, không thấy biểu đồ |
| ✅ **Tách `src/` (logic) + `notebooks/` (chạy + hiển thị)** | Tái sử dụng code, notebook ngắn gọn, dễ so sánh thí nghiệm | Phải import từ src |

**→ Đây là cách tổ chức chuẩn trong nghiên cứu Machine Learning.** Mỗi notebook là 1 "phase" của thí nghiệm, còn `src/` chứa code dùng chung cho tất cả phase.

### Vì sao tách thành 6 notebooks thay vì 1?

Mỗi notebook đại diện cho 1 **bước riêng biệt** trong quy trình nghiên cứu. Tách ra giúp:
- Chạy lại từng bước mà **không cần chạy lại từ đầu**
- Nếu bước train mất 3 tiếng, bạn không cần chạy lại khi chỉ muốn xem biểu đồ
- Dễ debug: biết lỗi ở bước nào

---

## 📦 File `requirements.txt`

```
torch, torchvision, numpy, pandas, matplotlib, scikit-learn, Pillow, scipy, pyyaml, jupyter
```

### Nó làm gì?
Liệt kê tất cả thư viện Python cần thiết để chạy project.

### Chạy thế nào?
```bash
pip install -r requirements.txt
```

### Tại sao cần file này?
Để bất kỳ ai clone project đều cài đúng thư viện. Không có file này → người khác phải đoán cần cài gì → lỗi `ModuleNotFoundError`.

### Tại sao lại chọn những thư viện này?

| Thư viện | Vai trò | Vì sao không dùng cái khác? |
|---|---|---|
| `torch` + `torchvision` | Framework deep learning chính | PyTorch phổ biến hơn TensorFlow trong nghiên cứu, có sẵn pretrained models |
| `numpy` | Xử lý mảng số | Thư viện nền tảng, gần như bắt buộc |
| `pandas` | Xử lý bảng dữ liệu | Tiện cho báo cáo kết quả |
| `matplotlib` | Vẽ biểu đồ | Chuẩn công nghiệp cho Python |
| `scikit-learn` | Confusion matrix, classification report | Không cần tự viết metrics |
| `Pillow` | Đọc/ghi ảnh | torchvision dùng Pillow bên dưới |
| `scipy` | Đọc file `.mat` (metadata dataset) | Oxford Flowers 102 dùng format MATLAB |

---

## 🗂️ Thư mục `flower_data/flower_data/`

### Nó chứa gì?
Dataset **Oxford Flowers 102** đã chia sẵn thành 3 phần:

```
flower_data/flower_data/
├── train/          # ~1,020 ảnh (10 ảnh/loại × 102 loại) - dùng để HỌC
│   ├── 1/          # Ảnh của loại hoa số 1
│   ├── 2/
│   └── ... (102 thư mục)
├── valid/          # ~1,020 ảnh - dùng để ĐÁNH GIÁ trong khi train
│   └── ...
├── test/           # ~6,149 ảnh - dùng để TEST CUỐI CÙNG
│   └── ...
└── cat_to_name.json  # {"1": "pink primrose", "2": "hard-leaved pocket orchid", ...}
```

### Tại sao chia 3 phần (train/valid/test)?

| Tập | Mục đích | Khi nào dùng? |
|---|---|---|
| **train** | Model **học** từ tập này | Mỗi epoch |
| **valid** | Đánh giá model **trong khi train** để chọn checkpoint tốt nhất | Sau mỗi epoch |
| **test** | Đánh giá model **lần cuối**, chỉ dùng 1 lần khi đã chọn xong model | Chỉ cuối cùng |

> ⚠️ **Nếu dùng test set để chọn model → kết quả bị thiên lệch (data leakage).** Đây là lỗi phổ biến nhất trong ML.

### Tại sao dùng format ImageFolder?

Đây là format mà `torchvision.datasets.ImageFolder` đọc trực tiếp:
```
train/
  1/     ← tên thư mục = label (class ID)
    image_001.jpg
    image_002.jpg
  2/
    ...
```
**Lợi ích**: Không cần viết code đọc label thủ công. PyTorch tự map tên thư mục → class index.

### Tại sao Oxford Flowers 102 mà không phải dataset khác?

| Dataset | Số class | Tổng ảnh | Độ khó | Phù hợp cho |
|---|---|---|---|---|
| CIFAR-10 | 10 | 60,000 | Dễ | Bài tập nhập môn |
| Oxford Flowers 102 | 102 | ~8,189 | **Khó** | Nghiên cứu fine-grained classification |
| ImageNet | 1,000 | 1.2M | Rất khó | Benchmark lớn (cần GPU mạnh) |

**Oxford Flowers 102 phù hợp** vì:
- 102 loại hoa → đủ thách thức (nhiều loại hoa rất giống nhau)
- Chỉ ~8K ảnh → chạy được trên GPU sinh viên (không cần cluster)
- **Chỉ có ~10 ảnh/loại để train** → buộc phải dùng transfer learning → đúng trọng tâm môn học

---

## 🔧 Source Code: `src/flowers102/`

### `__init__.py` — File khởi tạo module

```python
from .data import build_dataloaders, build_transforms, class_distribution
from .evaluate import evaluate_model, topk_accuracy
from .models import create_model
from .train import fit
from .utils import set_seed
```

**Nó làm gì?** Cho phép import gọn: `from flowers102.data import build_dataloaders` thay vì đường dẫn dài.

**Tại sao cần?** Không có file này → Python không nhận `flowers102` là 1 package → không import được.

---

### `models.py` — Quản lý tất cả model

#### Nó làm gì?
1. **Tạo model** pretrained từ ImageNet: `create_model("convnext_tiny", num_classes=102)`
2. **Thay classifier head**: Mỗi model có lớp cuối khác nhau, file này xử lý hết
3. **Freeze/Unfreeze backbone**: Cho 2-stage fine-tuning

#### Hỗ trợ 8 model:

| Model | Accuracy dự kiến | Vai trò | Classifier head |
|---|---|---|---|
| `vgg16` | ~71-79% | Kế thừa nhóm trước | `model.classifier[-1]` |
| `resnet18` | ~87-90% | Baseline nhẹ | `model.fc` |
| `resnet50` | ~88-93% | Kế thừa nhóm trước | `model.fc` |
| `efficientnet_b0` | ~92-95% | Baseline hiệu quả | `model.classifier[-1]` |
| `efficientnet_b3` | ~94-96% | Cải tiến CNN | `model.classifier[-1]` |
| `convnext_tiny` | ~96-97% | Modern CNN ⭐ | `model.classifier[2]` |
| `swin_t` | ~95-96% | Transformer | `model.head` |
| `vit_b_16` | ~95-97% | Vision Transformer | `model.heads.head` |

#### Vì sao cần `_replace_classifier`?

Mỗi model pretrained trên ImageNet có **1,000 output neurons** (1,000 loại trong ImageNet). Dự án chỉ có **102 loại hoa** → phải thay lớp cuối:
```
ImageNet head: Linear(in_features, 1000)  →  Hoa head: Linear(in_features, 102)
```

#### Vì sao mỗi model có classifier head khác nhau?

Đây là do **kiến trúc** của từng model khác nhau:
- **ResNet** dùng 1 layer `model.fc` (fully connected)
- **VGG** dùng `model.classifier` (Sequential với nhiều Linear layers)
- **EfficientNet** cũng dùng `model.classifier` nhưng cấu trúc khác VGG
- **ConvNeXt** dùng `model.classifier[2]` (LayerNorm → Flatten → Linear)
- **Swin** dùng `model.head` (1 Linear layer)
- **ViT** dùng `model.heads.head` (nested module)

→ File `models.py` **che giấu sự phức tạp này**: bạn chỉ cần gọi `create_model("tên")` mà không cần biết classifier nằm ở đâu.

#### Vì sao cần `freeze_feature_extractor`?

**2-Stage Fine-tuning** là kỹ thuật chuẩn:

```
Stage 1 (freeze backbone):
  - Backbone (đã học từ ImageNet) → ĐÓNG BĂNG, không thay đổi
  - Chỉ train classifier head (lớp cuối)
  - Lý do: head mới khởi tạo random, nếu train cả model → gradient lớn từ head
    sẽ phá hỏng features tốt của backbone

Stage 2 (unfreeze all):
  - Mở toàn bộ model
  - Fine-tune với learning rate nhỏ hơn
  - Lý do: head đã ổn định, giờ backbone có thể điều chỉnh nhẹ để phù hợp với hoa
```

**Nếu không freeze** → accuracy có thể thấp hơn 2-5% vì backbone bị "quên" features từ ImageNet.

#### Vì sao chọn ConvNeXt-Tiny mà không phải ViT-Large?

| Model | Params | GPU RAM cần | Accuracy | Dễ train? |
|---|---|---|---|---|
| ConvNeXt-Tiny | 28.6M | ~4GB | ~96-97% | ✅ Rất dễ |
| ViT-B/16 | 86.6M | ~8GB | ~95-97% | ⚠️ Cần tuning cẩn thận |
| ViT-L/16 | 305M | ~20GB | ~99% | ❌ Cần GPU mạnh |

**ConvNeXt-Tiny là lựa chọn tối ưu cho sinh viên** vì:
- Accuracy gần bằng ViT nhưng **nhẹ hơn 3x**
- **Ổn định hơn**: CNN truyền thống nên ít nhạy cảm với hyperparameter
- ViT cần warmup dài, differential LR → dễ train sai

---

### `data.py` — Xử lý dữ liệu và augmentation

#### Nó làm gì?
1. **Tạo data transforms** (tiền xử lý ảnh)
2. **Tạo DataLoader** (đọc ảnh theo batch)
3. **Mixup/CutMix collate** (augmentation ở mức batch)

#### Luồng xử lý 1 ảnh khi train:

```
Ảnh gốc (JPG, kích thước bất kỳ)
  ↓ Resize(256)              — co/giãn cạnh ngắn về 256px
  ↓ RandomResizedCrop(224)   — cắt ngẫu nhiên 1 vùng 224×224
  ↓ RandomHorizontalFlip     — lật ngang 50% xác suất
  ↓ RandAugment              — biến đổi ngẫu nhiên (xoay, đổi màu, blur, ...)
  ↓ ToTensor                 — chuyển sang tensor [0, 1]
  ↓ Normalize(ImageNet)      — chuẩn hóa theo mean/std của ImageNet
  ↓ RandomErasing (10%)      — xóa ngẫu nhiên 1 vùng hình chữ nhật
  = Tensor [3, 224, 224]     — sẵn sàng đưa vào model
```

#### Luồng xử lý 1 ảnh khi đánh giá (valid/test):

```
Ảnh gốc
  ↓ Resize(256)
  ↓ CenterCrop(224)          — cắt chính giữa (không random)
  ↓ ToTensor
  ↓ Normalize(ImageNet)
  = Tensor [3, 224, 224]
```

> **Khi đánh giá KHÔNG dùng augmentation** vì cần kết quả ổn định, có thể so sánh được.

#### Tại sao cần Data Augmentation?

Oxford Flowers 102 chỉ có **~10 ảnh/loại để train**. Quá ít! Augmentation "giả tạo" thêm ảnh:

```
1 ảnh gốc  →  Lật ngang      →  +1 ảnh
            →  Cắt vị trí khác →  +1 ảnh
            →  Xoay nhẹ        →  +1 ảnh
            →  Đổi độ sáng     →  +1 ảnh
            ...
```

Kết quả: Model thấy mỗi ảnh **ở nhiều góc nhìn khác nhau** → học được features tổng quát hơn → **+3-5% accuracy**.

#### Tại sao dùng RandAugment thay vì ColorJitter?

| Kỹ thuật | Cách hoạt động | Hiệu quả |
|---|---|---|
| `ColorJitter` | Thay đổi brightness, contrast, saturation, hue | Tốt nhưng giới hạn |
| **`RandAugment`** | Chọn ngẫu nhiên N phép biến đổi từ 14 phép (rotate, shear, posterize, equalize, ...) | **Mạnh hơn, đa dạng hơn** |

**RandAugment** mạnh hơn vì:
- Đa dạng: 14 loại biến đổi thay vì 4
- Tự động: chỉ cần tuning 2 tham số (`num_ops`, `magnitude`)
- Được chứng minh hiệu quả trong nhiều bài báo (Google Research, 2020)

#### Mixup và CutMix là gì?

```
Mixup (alpha=0.2):
  Ảnh mới = 0.8 × Ảnh_A + 0.2 × Ảnh_B
  Label mới = 0.8 × Label_A + 0.2 × Label_B
  → Model học "ảnh này 80% là hoa hồng, 20% là hoa cúc"

CutMix (alpha=1.0):
  Cắt 1 vùng hình chữ nhật từ Ảnh_B, dán vào Ảnh_A
  Label = tỉ lệ diện tích mỗi ảnh
  → Model phải nhận diện hoa từ từng phần nhỏ
```

**Tại sao cần?**
- **Chống overfitting**: Model không thể "thuộc lòng" 10 ảnh/loại
- **Accuracy tăng ~0.5-1.5%**
- CutMix đặc biệt tốt cho Flowers 102 vì buộc model nhận diện **từng bộ phận** của hoa (cánh, nhụy, lá) thay vì dựa vào background

#### Tại sao Normalize theo ImageNet?

```python
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
```

Model pretrained được train trên ImageNet với mean/std này. Nếu dùng mean/std khác → input distribution khác → features bị sai → accuracy giảm.

**Quy tắc: Luôn dùng cùng normalization với pretrained weights.**

---

### `train.py` — Vòng lặp huấn luyện

#### Nó làm gì?
1. **`_train_one_epoch()`**: Chạy 1 epoch (duyệt toàn bộ train set 1 lần)
2. **`fit()`**: Chạy nhiều epochs, lưu checkpoint, early stopping
3. **`build_scheduler_with_warmup()`**: Tạo learning rate schedule

#### Luồng 1 epoch:

```
Cho mỗi batch (32 ảnh):
  1. Đưa ảnh vào model → ra logits (102 số)
  2. Tính loss = CrossEntropy(logits, labels)
  3. Tính gradient (loss.backward)
  4. Cập nhật weights (optimizer.step)
  5. Đếm số dự đoán đúng

Sau epoch:
  6. Đánh giá trên validation set
  7. Nếu valid_top1 cao nhất → lưu checkpoint
  8. Nếu valid_top1 không cải thiện N epoch → dừng sớm (early stopping)
  9. Cập nhật learning rate (scheduler.step)
```

#### Khi chạy, terminal sẽ hiển thị:

```
[Epoch 1/20]  train_loss=2.3456  train_top1=0.4512  valid_loss=1.8765  valid_top1=0.5234  valid_top5=0.7890  lr=0.000050
[Epoch 2/20]  train_loss=1.5678  train_top1=0.6234  valid_loss=1.2345  valid_top1=0.6789  valid_top5=0.8901  lr=0.000100
...
[Epoch 15/20] train_loss=0.3456  train_top1=0.9123  valid_loss=0.4567  valid_top1=0.9345  valid_top5=0.9890  lr=0.000012
```

#### Tại sao dùng AMP (Automatic Mixed Precision)?

```
Không AMP: Tất cả tính toán dùng float32 (32-bit)
Có AMP:    Forward pass dùng float16 (16-bit), backward vẫn float32
```

| | Không AMP | Có AMP |
|---|---|---|
| GPU RAM | 6GB | ~3.5GB |
| Tốc độ | 1x | ~1.5-2x nhanh hơn |
| Accuracy | Chuẩn | Gần như bằng |

**→ Miễn phí tốc độ và tiết kiệm RAM, gần như không mất accuracy.**

#### Tại sao cần Gradient Clipping?

Khi gradient quá lớn (gradient explosion) → weights thay đổi quá mạnh → model "bùng nổ":

```
Không clip: gradient = 1000 → weight thay đổi rất lớn → loss = NaN
Có clip:    gradient = 1000 → bị cắt xuống 1.0 → weight thay đổi vừa phải
```

**Đặc biệt quan trọng cho Transformer models** (Swin, ViT) vì cơ chế attention dễ sinh gradient lớn.

#### Tại sao cần Warmup?

```
Không warmup:  Epoch 1 → lr = 0.0001 (quá lớn khi model chưa ổn định)
Có warmup:     Epoch 1 → lr = 0.000001 (rất nhỏ)
               Epoch 2 → lr = 0.000033
               Epoch 3 → lr = 0.000066
               Epoch 4 → lr = 0.0001   (đạt lr mong muốn)
               Epoch 5+ → giảm dần theo cosine
```

**Lý do**: Lúc đầu classifier head mới khởi tạo random → gradient rất nhiễu. Nếu lr lớn ngay → model bị "sốc" → features tốt từ ImageNet bị phá hỏng. Warmup cho model thời gian ổn định trước.

#### Tại sao dùng Cosine Annealing chứ không phải Step LR?

```
Step LR:    lr giảm đột ngột tại epoch 10, 20 → model bị "giật"
Cosine:     lr giảm mượt theo đường cong cosine → model hội tụ ổn định hơn
```

```
Cosine LR Schedule:
lr │ ╲
   │   ╲
   │    ╲
   │     ╲
   │      ╲_____
   └──────────── epoch
```

**+0.5-1% accuracy so với Step LR** trên hầu hết bài toán.

#### Tại sao cần Early Stopping?

```
Epoch 10: valid_top1 = 0.94  ← tốt nhất → lưu checkpoint
Epoch 11: valid_top1 = 0.93  ← giảm, wait = 1
Epoch 12: valid_top1 = 0.92  ← giảm tiếp, wait = 2
...
Epoch 17: valid_top1 = 0.90  ← wait = 7 ≥ patience → DỪNG
```

**Lý do**: Khi valid accuracy không tăng nữa → model đang overfitting (train accuracy vẫn tăng nhưng valid giảm). Tiếp tục train chỉ phí thời gian và model càng tệ hơn.

---

### `evaluate.py` — Đánh giá model

#### Nó làm gì?
1. **`topk_accuracy()`**: Tính Top-K accuracy
2. **`evaluate_model()`**: Chạy model trên 1 dataset, trả về loss + top1 + top5

#### Top-1 vs Top-5 nghĩa là gì?

```
Model dự đoán 5 loại hoa xác suất cao nhất:
  1. Sunflower   (60%)  ← Top-1 prediction
  2. Daisy       (20%)
  3. Dandelion   (10%)
  4. Buttercup    (5%)
  5. Marigold     (5%)

Đáp án đúng: Daisy

→ Top-1: SAI (vì dự đoán #1 là Sunflower, không phải Daisy)
→ Top-5: ĐÚNG (vì Daisy nằm trong top 5)
```

**Tại sao cần cả Top-5?**
- Nhiều loại hoa rất giống nhau (ví dụ: các loại daisy)
- Top-5 cho thấy model "gần đúng" mặc dù Top-1 sai
- Target: **Top-1 ≥ 90%, Top-5 ≥ 97%**

#### Khi chạy ra gì?

```python
{"loss": 0.4567, "top1": 0.9345, "top5": 0.9890}
```

→ Model đúng 93.45% (Top-1) và 98.90% (Top-5)

---

### `experiment.py` — Chạy thí nghiệm từ command line

#### Nó làm gì?
Cho phép chạy baseline training bằng 1 lệnh terminal mà không cần mở notebook:

```bash
python -m src.flowers102.experiment --mode baseline --data-dir flower_data/flower_data --out-dir .
```

#### Khi chạy ra gì?
- `checkpoints/baseline_best.pth` — weights model tốt nhất
- `reports/baseline_history.json` — lịch sử loss/accuracy mỗi epoch
- `reports/baseline_test_metrics.json` — kết quả test cuối cùng

#### Tại sao cần khi đã có notebook?
- **Notebook**: chạy từng bước, xem biểu đồ, interactive
- **CLI**: chạy tự động, có thể đặt script chạy ban đêm, tích hợp vào pipeline

---

### `utils.py` — Các hàm tiện ích

#### Nó làm gì?
1. **`set_seed(42)`**: Đặt random seed cho reproducibility
2. **`ensure_dir()`**: Tạo thư mục nếu chưa tồn tại
3. **`save_json()`**: Lưu dictionary ra file JSON

#### Tại sao seed = 42?

Khi đặt seed, mọi phép random (khởi tạo weights, shuffle data, augmentation) đều cho **cùng kết quả mỗi lần chạy**. Không có seed → mỗi lần chạy ra kết quả khác → không thể so sánh thí nghiệm.

**42 là convention** (Douglas Adams' "The Hitchhiker's Guide to the Galaxy"). Có thể dùng bất kỳ số nào, miễn là cố định.

---

## ⚙️ Configs: `configs/`

### `baseline.yaml`

```yaml
model_name: efficientnet_b0   # Model dùng cho baseline
epochs: 12                     # Số epoch train
optimizer:
  name: adamw
  lr: 0.0003                   # Learning rate
  weight_decay: 0.0001         # Regularization (chống overfitting)
scheduler:
  name: cosine
  t_max: 12                    # Cosine annealing period = số epoch
```

**Tại sao dùng YAML?**
- Dễ đọc hơn JSON (không cần dấu ngoặc kép, hỗ trợ comment)
- Tách config ra khỏi code → đổi hyperparameter mà **không sửa code**
- Dễ quản lý nhiều thí nghiệm: chỉ cần copy file YAML, đổi vài dòng

### `advanced.yaml`

```yaml
models:                        # 5 model cần so sánh
  - vgg16                      # ← kế thừa
  - resnet50                   # ← kế thừa
  - efficientnet_b3            # ← cải tiến
  - convnext_tiny              # ← cải tiến (mạnh nhất)
  - swin_t                     # ← cải tiến (transformer)

stage1:                        # Stage 1: freeze backbone
  freeze_backbone: true
  epochs: 5
  lr: 0.0005

stage2:                        # Stage 2: full fine-tune
  unfreeze_all: true
  epochs: 20
  lr: 0.0001
  warmup_epochs: 3
  grad_clip_max_norm: 1.0
  use_amp: true

regularization:
  label_smoothing: 0.1         # Làm mềm label (chống overconfident)
  mixup_alpha: 0.2             # Trộn ảnh
  cutmix_alpha: 1.0            # Cắt-dán ảnh

augmentation:
  randaugment: true            # Dùng RandAugment thay ColorJitter
  random_erasing: 0.1          # 10% xóa ngẫu nhiên 1 vùng
```

### `search_space.yaml`

Định nghĩa **không gian tìm kiếm hyperparameter** cho notebook 04:

```yaml
search:
  lr: [0.00003 ... 0.001]      # Thử nhiều learning rate
  model_name: [resnet50, efficientnet_b3, convnext_tiny, swin_t]
  label_smoothing: [0.0, 0.05, 0.1]
```

**Tại sao cần?** Không ai biết trước hyperparameter nào tốt nhất. Random search thử nhiều tổ hợp → tìm ra tổ hợp tốt nhất.

---

## 📓 Notebooks: `notebooks/`

### `01_data_audit.ipynb` — Kiểm tra dữ liệu

#### Nó làm gì?
1. Đếm số ảnh mỗi split (train/valid/test)
2. Kiểm tra class balance (mỗi loại có bao nhiêu ảnh)
3. Kiểm tra ảnh bị hỏng (corrupted)
4. Kiểm tra data leakage (ảnh trùng giữa train và test)

#### Khi chạy ra gì?

```
Split    | num_images | num_classes
train    | 1020       | 102
valid    | 1020       | 102
test     | 6149       | 102

min/max class count: 10 10
duplicate train-valid: 0
duplicate train-test: 0
```

Output: `reports/data_audit_report.json`

#### Tại sao cần?

**Trước khi train, PHẢI kiểm tra dữ liệu.** Những vấn đề có thể gặp:
- Ảnh bị hỏng → model crash khi load
- Ảnh trùng giữa train và test → accuracy cao giả (data leakage)
- Class imbalance → model thiên vị loại hoa có nhiều ảnh

**Nếu không kiểm tra → có thể train xong mới phát hiện lỗi → mất thời gian.**

---

### `02_baseline_train.ipynb` — Huấn luyện baseline

#### Nó làm gì?
Train **EfficientNet-B0** (pretrained ImageNet) với recipe đơn giản:
- 12 epochs, AdamW, lr=3e-4
- Cosine annealing, early stopping patience=4
- Augmentation cơ bản (crop, flip, ColorJitter)

#### Khi chạy ra gì?

```
[Epoch 1/12]  train_loss=3.12  train_top1=0.21  valid_loss=2.45  valid_top1=0.35  ...
[Epoch 2/12]  train_loss=2.01  train_top1=0.48  valid_loss=1.67  valid_top1=0.55  ...
...
[Epoch 12/12] train_loss=0.45  train_top1=0.88  valid_loss=0.52  valid_top1=0.92  ...
```

Biểu đồ Loss và Top-1 theo epoch.

Output:
- `checkpoints/baseline_best.pth` — model checkpoint
- `reports/baseline_history.json` — training history

#### Tại sao EfficientNet-B0 làm baseline?

| Model | Params | Accuracy | Tốc độ train |
|---|---|---|---|
| ResNet18 | 11.7M | ~87% | Nhanh nhưng accuracy thấp |
| **EfficientNet-B0** | **5.3M** | **~92-95%** | **Nhanh + accuracy tốt** |
| ResNet50 | 25.6M | ~91% | Chậm hơn mà accuracy thấp hơn B0 |

EfficientNet-B0 **nhẹ nhất nhưng accuracy cao nhất** → baseline lý tưởng.

---

### `03_advanced_training.ipynb` — So sánh 5 model ⭐ (QUAN TRỌNG NHẤT)

#### Nó làm gì?
1. Áp dụng toàn bộ kỹ thuật cải tiến (RandAugment, Mixup/CutMix, Warmup, AMP, ...)
2. Train lần lượt 5 model với **cùng recipe, cùng data**
3. So sánh kết quả trong bảng và biểu đồ

#### Khi chạy ra gì?

**Bảng so sánh** (giá trị dự kiến):

```
╔══════════════════╦═══════════╦═══════════════╦═══════════════╗
║ Model            ║ Type      ║ Test Top-1 (%)║ Test Top-5 (%)║
╠══════════════════╬═══════════╬═══════════════╬═══════════════╣
║ convnext_tiny    ║ Cải tiến  ║ 96.50         ║ 99.20         ║
║ swin_t           ║ Cải tiến  ║ 95.80         ║ 99.10         ║
║ efficientnet_b3  ║ Cải tiến  ║ 95.20         ║ 98.80         ║
║ resnet50         ║ Kế thừa   ║ 92.30         ║ 98.10         ║
║ vgg16            ║ Kế thừa   ║ 78.50         ║ 93.20         ║
╚══════════════════╩═══════════╩═══════════════╩═══════════════╝
```

**Biểu đồ**:
- Bar chart so sánh accuracy (đỏ = kế thừa, xanh = cải tiến)
- Training curves cho từng model

Output:
- `reports/advanced_all_results.json`
- `reports/model_comparison.png`
- `reports/training_curves.png`
- `reports/final_comparison_report.json`
- `checkpoints/advanced/{model}_best.pth` — checkpoint cho mỗi model

#### Tại sao đây là notebook quan trọng nhất?

Đây là notebook trả lời trực tiếp yêu cầu của thầy:
- **Kế thừa**: VGG16 và ResNet50 từ nhóm trước
- **Cải tiến**: Thêm 3 model mạnh hơn + training recipe tốt hơn
- **So sánh**: Cùng điều kiện, rõ ràng model nào tốt hơn và tốt hơn bao nhiêu

---

### `04_hparam_search.ipynb` — Tìm kiếm hyperparameter

#### Nó làm gì?
Random search: thử ngẫu nhiên nhiều tổ hợp hyperparameter, chọn tổ hợp tốt nhất.

Mỗi trial chỉ train 4 epochs (nhanh) để **ước lượng** model nào + config nào có tiềm năng.

#### Khi chạy ra gì?

```
trial 0: model=resnet50,    lr=3e-4, valid_top1=0.72
trial 1: model=convnext_tiny, lr=1e-4, valid_top1=0.81  ← tốt nhất
trial 2: model=efficientnet_b3, lr=5e-4, valid_top1=0.78
...
```

Output: `reports/hparam_search_results.json`

#### Tại sao dùng Random Search chứ không phải Grid Search?

```
Grid Search:   Thử TẤT CẢ tổ hợp → 3 model × 3 lr × 2 wd × 2 bs × 3 ls = 108 trials
Random Search: Thử 8-20 tổ hợp NGẪU NHIÊN
```

**Random search hiệu quả hơn** (Bergstra & Bengio, 2012):
- Grid search lãng phí thời gian cho tham số không quan trọng
- Random search khám phá không gian rộng hơn với cùng số trial
- Với 20 trials, xác suất tìm được config trong top-5% là 64%

---

### `05_evaluation_error_analysis.ipynb` — Phân tích lỗi

#### Nó làm gì?
1. Load model tốt nhất từ checkpoint
2. Chạy trên test set → tính metrics cuối cùng
3. Vẽ **Confusion Matrix** (ma trận nhầm lẫn)
4. Xuất **Classification Report** (precision, recall, F1 từng loại)

#### Khi chạy ra gì?

**Confusion Matrix**: Ma trận 102×102, cho thấy model hay nhầm loại nào với loại nào.

**Classification Report**:
```
class 1 (pink primrose):    precision=0.95  recall=0.92  f1=0.93
class 2 (hard orchid):      precision=0.88  recall=0.85  f1=0.86  ← yếu hơn
...
```

Output: `reports/final_evaluation.json`

#### Tại sao cần phân tích lỗi?

Biết accuracy = 93% là không đủ. Cần biết:
- Model hay sai ở loại nào? (→ cần augmentation riêng cho loại đó)
- Model nhầm loại nào với loại nào? (→ 2 loại đó có thể rất giống nhau)
- Có pattern nào? (ví dụ: model yếu ở hoa nhỏ, hoặc hoa có background phức tạp)

---

### `06_inference_demo.ipynb` — Demo dự đoán

#### Nó làm gì?
Lấy 1 ảnh → đưa vào model → hiển thị top-5 dự đoán.

#### Khi chạy ra gì?

```
sample: flower_data/flower_data/train/1/image_06734.jpg

1  pink primrose    0.9523
2  gazania          0.0234
3  osteospermum     0.0112
4  mexican aster    0.0067
5  barbeton daisy   0.0031
```

#### Tại sao cần notebook này?

- **Trực quan**: Cho thầy/người xem thấy model hoạt động thực tế
- **Kiểm tra nhanh**: Đưa ảnh bất kỳ vào xem model dự đoán có đúng không
- **Demo**: Nếu phải present → chạy notebook này live

---

## 📊 Thư mục output: `checkpoints/` và `reports/`

### `checkpoints/`
Chứa file `.pth` — **trạng thái model** (weights, biases). Dùng để:
- Load lại model mà **không cần train lại**
- So sánh model ở các thời điểm khác nhau
- Deploy model vào ứng dụng thực tế

### `reports/`
Chứa JSON + PNG — **kết quả thí nghiệm**. Dùng để:
- Viết báo cáo
- So sánh các thí nghiệm
- Tái tạo kết quả (reproducibility)

---

## 🔄 Quy trình chạy từ đầu đến cuối

```
Bước 1: Cài thư viện
  pip install -r requirements.txt

Bước 2: Kiểm tra dữ liệu
  Chạy notebooks/01_data_audit.ipynb
  → Xác nhận data đúng, không bị hỏng, không leakage

Bước 3: Train baseline
  Chạy notebooks/02_baseline_train.ipynb
  → Có kết quả EfficientNet-B0 (~92-95%) làm mốc so sánh

Bước 4: Train và so sánh 5 model ⭐
  Chạy notebooks/03_advanced_training.ipynb
  → Bảng so sánh VGG16, ResNet50 vs EfficientNet-B3, ConvNeXt, Swin
  → ⏱ Mất ~2-4 tiếng tùy GPU

Bước 5 (optional): Tìm hyperparameter tối ưu
  Chạy notebooks/04_hparam_search.ipynb
  → Tìm config tốt nhất cho model tốt nhất

Bước 6: Phân tích lỗi
  Chạy notebooks/05_evaluation_error_analysis.ipynb
  → Confusion matrix, biết model yếu ở loại nào

Bước 7: Demo
  Chạy notebooks/06_inference_demo.ipynb
  → Demo dự đoán 1 ảnh
```

---

## 🔬 Batch Normalization, Dropout, Skip Connection — Có áp dụng không?

**Trả lời ngắn: CÓ — nhưng chúng đã có sẵn bên trong model pretrained, không cần thêm thủ công.**

Khi dùng `create_model("convnext_tiny", pretrained=True)`, bạn nhận được model **đã bao gồm tất cả kỹ thuật** bên trong kiến trúc. Đây là bảng chi tiết:

### Bảng so sánh kỹ thuật có sẵn trong từng model

| Kỹ thuật | VGG16 | ResNet50 | EfficientNet-B3 | ConvNeXt-Tiny |
|---|---|---|---|---|
| **Batch Normalization** | ❌ Không có | ✅ Có (`BatchNorm2d` sau mỗi Conv) | ✅ Có (`BatchNorm2d` trong MBConv) | ❌ Dùng `LayerNorm` (tốt hơn BN) |
| **Dropout** | ✅ Có (`Dropout(0.5)` ở classifier) | ❌ Không có | ✅ Có (`Dropout(0.3)` ở classifier) | ❌ Dùng `StochasticDepth` (tốt hơn Dropout) |
| **Skip Connection** | ❌ Không có | ✅ Có (residual connection mỗi block) | ✅ Có (residual trong MBConv) | ✅ Có (residual trong CNBlock) |
| **Stochastic Depth** | ❌ | ❌ | ✅ Có | ✅ Có |
| **Squeeze-Excitation** | ❌ | ❌ | ✅ Có | ❌ |

### Chi tiết kiến trúc thực tế (output từ PyTorch)

#### ConvNeXt-Tiny — 1 block bên trong:
```
CNBlock(
  Conv2d(768, 768, kernel_size=(7,7), groups=768)  ← depthwise conv 7×7
  LayerNorm((768,))                                 ← LayerNorm (thay cho BatchNorm)
  Linear(768 → 3072)                                ← expand
  GELU()                                            ← activation mới (mượt hơn ReLU)
  Linear(3072 → 768)                                ← compress
  StochasticDepth(p=0.088)                          ← thay cho Dropout
)
+ residual connection (skip connection)              ← có sẵn
```

#### EfficientNet-B3 — 1 block bên trong:
```
MBConv(
  Conv2d(24 → 144, 1×1) + BatchNorm2d + SiLU       ← expand + BN + activation
  Conv2d(144, 3×3, groups=144) + BatchNorm2d + SiLU ← depthwise conv + BN
  SqueezeExcitation(avgpool → fc1 → fc2 → sigmoid)  ← attention cơ chế
  Conv2d(144 → 32, 1×1) + BatchNorm2d               ← compress + BN
  StochasticDepth(p=0.015)                           ← drop path
)
+ residual connection (skip connection)              ← có sẵn
```

#### EfficientNet-B3 — Classifier head:
```
Sequential(
  Dropout(p=0.3)           ← Dropout ĐÃ CÓ SẴN
  Linear(1536 → 102)       ← chỉ thay output từ 1000 → 102
)
```

### Vì sao KHÔNG cần thêm Dropout/BN thủ công?

**1. Model pretrained đã được thiết kế tối ưu:**
   - Các tác giả (Google, Facebook Research) đã thử nghiệm hàng trăm tổ hợp → kiến trúc release là tốt nhất
   - Thêm Dropout vào giữa backbone → **phá vỡ pretrained weights** → accuracy giảm

**2. Thêm Dropout vào classifier head?**
   - EfficientNet-B3 **đã có** `Dropout(0.3)` ở classifier → code của mình tự giữ lại khi thay `classifier[-1]`
   - ConvNeXt-Tiny dùng `StochasticDepth` (drop toàn bộ block thay vì drop từng neuron) → **hiệu quả hơn Dropout** cho deep network

**3. Thêm BatchNorm?**
   - ConvNeXt đã chủ động **thay BatchNorm bằng LayerNorm** vì LayerNorm ổn định hơn với batch size nhỏ (batch=32 trong project này)
   - Thêm BN vào model đã dùng LayerNorm → xung đột normalization → accuracy giảm

**4. Thêm Skip Connection?**
   - Cả EfficientNet-B3 và ConvNeXt-Tiny **đã có skip connection** trong mỗi block
   - Thêm skip connection bên ngoài không có ý nghĩa

### Vậy mình cải tiến ở đâu?

Thay vì sửa **bên trong** model (đã tối ưu), mình cải tiến ở **bên ngoài** (training recipe):

| Kỹ thuật | Thuộc về | Vì sao hiệu quả? |
|---|---|---|
| **Mixup + CutMix** | Data-level regularization | Mạnh hơn Dropout cho image classification (+1-2%) |
| **Label Smoothing** | Loss-level regularization | Chống overconfident, bổ sung cho Dropout |
| **RandAugment** | Data augmentation | Tăng đa dạng dữ liệu, giảm overfitting |
| **Warmup + Cosine LR** | Optimizer | Bảo vệ pretrained features, hội tụ tốt hơn |
| **Gradient Clipping** | Training stability | Ngăn gradient explosion |
| **AMP** | Compute efficiency | Train nhanh 2x, tiết kiệm GPU RAM |

**Kết luận**: Dự án này áp dụng đầy đủ BatchNorm/LayerNorm, Dropout/StochasticDepth, Skip Connection — tất cả **có sẵn trong model pretrained**. Cải tiến của nhóm mình tập trung vào **training recipe** (cách train) thay vì sửa kiến trúc model (đã tối ưu).

---

## ❓ FAQ — Câu hỏi thường gặp

### Q: Chạy trên CPU được không?
**Được**, nhưng rất chậm. Notebook 03 (5 model) có thể mất **8-12 tiếng** trên CPU, so với **2-4 tiếng** trên GPU.

### Q: GPU RAM không đủ thì sao?
Giảm `batch_size` từ 32 xuống 16 hoặc 8. Nếu vẫn không đủ, bỏ Swin-T và ViT (cần nhiều RAM nhất).

### Q: Tại sao không dùng TensorFlow?
PyTorch phổ biến hơn trong nghiên cứu, có `torchvision` tích hợp sẵn pretrained models và transforms. TensorFlow cũng được, nhưng cộng đồng CV nghiêng về PyTorch.

### Q: Kết quả mỗi lần chạy có giống nhau không?
**Có**, nhờ `set_seed(42)`. Tất cả random operations (weight init, data shuffle, augmentation) đều deterministic.

### Q: Tại sao model kế thừa (VGG16) accuracy thấp thế?
VGG16 có **138M parameters** nhưng ImageNet accuracy chỉ **71.6%** — kiến trúc cũ (2014), không có skip connections, BatchNorm. Với chỉ 10 ảnh/loại → classifier khổng lồ (4096 neurons) dễ bị overfitting nghiêm trọng.
