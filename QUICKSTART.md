# Quick Start Guide - Sign2VN

Hướng dẫn nhanh để bắt đầu với Sign2VN trong vòng 10 phút.

## Bước 1: Setup Google Colab (2 phút)

1. Mở Google Colab: https://colab.research.google.com/
2. Tạo notebook mới hoặc upload `Sign2VN_Training.ipynb`
3. Chọn Runtime → Change runtime type → GPU (T4)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

## Bước 2: Upload Code (1 phút)

**Option A: Clone từ GitHub**
```bash
!git clone https://github.com/your-repo/sign2vn.git
%cd sign2vn
```

**Option B: Upload files thủ công**
- Upload tất cả các file `.py` vào `/content/sign2vn/`

## Bước 3: Cài Đặt Dependencies (2 phút)

```bash
!pip install -q torch torchvision
!pip install -q mediapipe opencv-python
!pip install -q nltk tqdm pandas scikit-learn

import nltk
nltk.download('punkt')
```

## Bước 4: Chuẩn Bị Dữ Liệu (0 phút - đã có sẵn)

Đảm bảo structure như sau:
```
MyDrive/Sign2VN/
├── meta.csv
└── work/landmarks/*.npy
```

## Bước 5: Training (5 phút setup)

```python
%cd /content/sign2vn

# Training với settings mặc định
!python train.py --num_epochs 100 --batch_size 16 --test
```

Hoặc dùng Python API:

```python
import sys
sys.path.append('/content/sign2vn')

from data_loader import VietnameseTokenizer, load_and_split_data, create_dataloaders
from model import Sign2TextModel
from trainer import Trainer
import config

# 1. Load data
tokenizer = VietnameseTokenizer()
train_dataset, val_dataset, test_dataset = load_and_split_data(
    config.META_CSV_PATH, tokenizer
)
train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset, val_dataset, test_dataset
)

# 2. Create model
model = Sign2TextModel(vocab_size=tokenizer.vocab_size)

# 3. Train
trainer = Trainer(model, tokenizer, train_loader, val_loader)
trainer.train(num_epochs=100)

# 4. Test
trainer.test(test_loader)
```

## Bước 6: Monitor Training

Training sẽ tự động:
- ✅ Save checkpoints mỗi epoch
- ✅ Early stopping nếu không improve
- ✅ Giảm learning rate tự động
- ✅ Print metrics mỗi epoch

Output mẫu:
```
Epoch 10 Summary:
  Train Loss: 1.2345 | Train Acc: 0.7890
  Val Loss:   1.1234 | Val Acc:   0.8123 | Val BLEU: 0.7456
  ✓ New best model!
```

## Bước 7: Inference

### Từ file .npy:
```python
from inference import SignLanguagePredictor

predictor = SignLanguagePredictor(
    checkpoint_path="/content/drive/MyDrive/Sign2VN/checkpoints/best_model.pt",
    tokenizer_path="/content/drive/MyDrive/Sign2VN/checkpoints/tokenizer.pkl"
)

result = predictor.predict_from_npy("path/to/landmarks.npy")
print(f"Prediction: {result['text']}")
```

### Từ video:
```python
result = predictor.predict_from_video("path/to/video.mp4", save_npy=True)
print(f"Prediction: {result['text']}")
```

### Batch prediction:
```python
results = predictor.batch_predict_from_folder(
    folder_path="/path/to/folder",
    file_extension=".npy"
)

for r in results:
    print(f"{r['filename']}: {r['text']}")
```

## Bước 8: Visualize Results

```python
from visualization import plot_training_curves
import json

# Load history
with open('/content/drive/MyDrive/Sign2VN/checkpoints/training_history.json', 'r') as f:
    history = json.load(f)

# Plot
plot_training_curves(history)
```

## Common Issues & Solutions

### ❌ CUDA Out of Memory
```python
# Giảm batch size
!python train.py --batch_size 8
```

### ❌ File not found
```python
# Kiểm tra đường dẫn
import os
print(os.path.exists('/content/drive/MyDrive/Sign2VN/meta.csv'))
```

### ❌ Model không học
```python
# Thử learning rate nhỏ hơn
!python train.py --learning_rate 0.0001
```

## Tips cho Kết Quả Tốt

1. **Dữ liệu**: Càng nhiều càng tốt (>1000 samples)
2. **Training time**: Ít nhất 50-100 epochs
3. **Validation**: Luôn monitor val_loss
4. **Best model**: Dùng best_model.pt chứ không phải latest
5. **Patience**: Training có thể mất vài giờ!

## Next Steps

Sau khi training xong:

1. ✅ Test trên test set
2. ✅ Visualize attention weights
3. ✅ Try inference trên video mới
4. ✅ Fine-tune hyperparameters
5. ✅ Collect more data nếu cần

## Cheat Sheet

```bash
# Training
python train.py --num_epochs 100 --batch_size 16

# Resume training
python train.py --resume_from checkpoints/latest_checkpoint.pt

# Inference single file
python inference.py --npy_path file.npy

# Inference video
python inference.py --video_path video.mp4 --save_npy

# Batch inference
python inference.py --folder_path /path/to/folder --file_extension .npy
```

---

**Tổng thời gian: ~10 phút setup + vài giờ training**

Good luck! 🚀
