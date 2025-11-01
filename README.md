# Sign2VN - Chuyển Đổi Ngôn Ngữ Ký Hiệu Sang Tiếng Việt

Hệ thống AI sử dụng Deep Learning (CNN + LSTM + Seq2Seq with Attention) để nhận diện ngôn ngữ ký hiệu từ video và chuyển đổi thành văn bản tiếng Việt tự nhiên.

## 📋 Mục Lục

- [Tính Năng](#-tính-năng)
- [Kiến Trúc Model](#-kiến-trúc-model)
- [Yêu Cầu Hệ Thống](#-yêu-cầu-hệ-thống)
- [Cài Đặt](#-cài-đặt)
- [Cấu Trúc Dữ Liệu](#-cấu-trúc-dữ-liệu)
- [Sử Dụng](#-sử-dụng)
- [Cấu Hình](#-cấu-hình)
- [Kết Quả](#-kết-quả)
- [Tips & Best Practices](#-tips--best-practices)

## ✨ Tính Năng

- ✅ **Training từ đầu** hoặc **resume** từ checkpoint
- ✅ **Seq2Seq với Attention** để tạo câu tiếng Việt tự nhiên
- ✅ **Data augmentation**: Gaussian noise, time warping, horizontal flip
- ✅ **Early stopping** và **learning rate scheduling**
- ✅ **Metrics đầy đủ**: Loss, Accuracy, BLEU score
- ✅ **Inference** từ video mới hoặc file .npy
- ✅ **Batch prediction** cho nhiều files
- ✅ **Visualization** training history
- ✅ **Checkpointing** tự động

## 🏗️ Kiến Trúc Model

```
Input Video → MediaPipe Landmarks → Model → Vietnamese Text

Model Architecture:
┌─────────────────────────────────────────────────────────┐
│  Input: Landmarks (T, 543*3)                            │
│     ↓                                                    │
│  Spatial Encoder (CNN)                                  │
│     - Conv1D layers với BatchNorm                       │
│     - Extract spatial features từ landmarks             │
│     ↓                                                    │
│  Temporal Encoder (Bidirectional LSTM)                  │
│     - 2 layers LSTM                                     │
│     - Capture temporal dependencies                     │
│     ↓                                                    │
│  Seq2Seq Decoder với Attention                          │
│     - Bahdanau Attention mechanism                      │
│     - LSTM decoder với teacher forcing                  │
│     - Embedding layer cho output tokens                 │
│     ↓                                                    │
│  Output: Vietnamese Text                                │
└─────────────────────────────────────────────────────────┘
```

### Components Chi Tiết:

1. **Spatial Encoder (CNN)**:
   - Input: `(batch, seq_len, 543*3)` - landmarks từ MediaPipe
   - Conv1D layers: `[64, 128, 256]` filters
   - BatchNorm và Dropout cho regularization
   - Output: `(batch, seq_len, 256)` spatial features

2. **Temporal Encoder (Bidirectional LSTM)**:
   - Input: Spatial features
   - 2-layer Bi-LSTM với 512 hidden units
   - Output: `(batch, seq_len, 1024)` temporal features

3. **Attention Mechanism**:
   - Bahdanau attention để focus vào relevant frames
   - Attention dimension: 256
   - Dynamic weighting của encoder outputs

4. **Seq2Seq Decoder**:
   - Embedding layer: 256 dimensions
   - LSTM decoder: 512 hidden units
   - Teacher forcing during training
   - Greedy decoding during inference

## 💻 Yêu Cầu Hệ Thống

- Python 3.8+
- CUDA-capable GPU (khuyến nghị, có thể chạy CPU nhưng chậm)
- Google Colab (khuyến nghị cho training)
- Google Drive với ít nhất 5GB trống

## 📦 Cài Đặt

### 1. Trên Google Colab (Khuyến Nghị)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository hoặc upload files
!git clone https://github.com/your-repo/sign2vn.git
# Hoặc upload các files .py vào Colab

# Cài đặt dependencies
!pip install -r requirements.txt

# Download NLTK data
import nltk
nltk.download('punkt')
```

### 2. Local Installation

```bash
# Clone repository
git clone https://github.com/your-repo/sign2vn.git
cd sign2vn

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## 📂 Cấu Trúc Dữ Liệu

Dữ liệu cần được tổ chức như sau trên Google Drive:

```
MyDrive/Sign2VN/
├── meta.csv                    # Metadata file
├── work/
│   └── landmarks/              # Thư mục chứa file .npy
│       ├── video1.npy
│       ├── video2.npy
│       └── ...
├── checkpoints/                # Sẽ được tạo tự động
│   ├── best_model.pt
│   ├── latest_checkpoint.pt
│   ├── tokenizer.pkl
│   └── training_history.json
└── logs/                       # Logs (optional)
```

### Format của `meta.csv`:

```csv
npy,label,label_vi,orig_name,signer
/path/to/file1.npy,LABEL1,tiếng việt 1,video1.mp4,UNKNOWN
/path/to/file2.npy,LABEL2,tiếng việt 2,video2.mp4,UNKNOWN
```

### Format của file `.npy`:

- Shape: `(num_frames, 543*3)`
- `543` landmarks = 33 pose + 21 left_hand + 21 right_hand + 468 face
- Mỗi landmark có 3 coordinates: `(x, y, z)`

## 🚀 Sử Dụng

### 1. Training Model

#### Sử dụng Colab Notebook (Dễ nhất):

1. Upload `Sign2VN_Training.ipynb` lên Google Colab
2. Mount Google Drive
3. Chạy các cells theo thứ tự

#### Sử dụng Command Line:

```bash
# Training từ đầu
python train.py \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --test

# Resume từ checkpoint
python train.py \
    --resume_from /path/to/checkpoint.pt \
    --num_epochs 100

# Custom settings
python train.py \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --checkpoint_dir /path/to/checkpoints \
    --test
```

#### Parameters:

- `--num_epochs`: Số epochs training (default: 100)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 0.001)
- `--checkpoint_dir`: Thư mục lưu checkpoints
- `--resume_from`: Path đến checkpoint để resume
- `--test`: Chạy test sau khi training xong
- `--seed`: Random seed (default: 42)

### 2. Inference - Dự Đoán

#### Từ file .npy:

```bash
python inference.py \
    --npy_path /path/to/landmarks.npy \
    --checkpoint /path/to/best_model.pt \
    --tokenizer /path/to/tokenizer.pkl
```

#### Từ video (sẽ tự động extract landmarks):

```bash
python inference.py \
    --video_path /path/to/video.mp4 \
    --save_npy \
    --checkpoint /path/to/best_model.pt \
    --tokenizer /path/to/tokenizer.pkl
```

#### Batch prediction:

```bash
python inference.py \
    --folder_path /path/to/folder \
    --file_extension .npy \
    --output predictions.json \
    --checkpoint /path/to/best_model.pt \
    --tokenizer /path/to/tokenizer.pkl
```

#### Sử dụng Python API:

```python
from inference import SignLanguagePredictor

# Khởi tạo predictor
predictor = SignLanguagePredictor(
    checkpoint_path="/path/to/best_model.pt",
    tokenizer_path="/path/to/tokenizer.pkl"
)

# Dự đoán từ .npy
result = predictor.predict_from_npy("landmarks.npy")
print(f"Prediction: {result['text']}")

# Dự đoán từ video
result = predictor.predict_from_video("video.mp4", save_npy=True)
print(f"Prediction: {result['text']}")

# Batch prediction
results = predictor.batch_predict_from_folder(
    folder_path="/path/to/folder",
    file_extension=".npy"
)
```

### 3. Visualization

```python
import json
import matplotlib.pyplot as plt

# Load training history
with open('checkpoints/training_history.json', 'r') as f:
    history = json.load(f)

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## ⚙️ Cấu Hình

Tất cả cấu hình được định nghĩa trong `config.py`. Các tham số quan trọng:

### Model Hyperparameters:

```python
# CNN
CNN_FILTERS = [64, 128, 256]
CNN_KERNEL_SIZE = 3

# LSTM
LSTM_UNITS = 512
LSTM_LAYERS = 2

# Seq2Seq
ENCODER_HIDDEN_DIM = 512
DECODER_HIDDEN_DIM = 512
ATTENTION_DIM = 256
EMBEDDING_DIM = 256

# Regularization
DROPOUT_RATE = 0.3
```

### Training Settings:

```python
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7
```

### Data Augmentation:

```python
USE_AUGMENTATION = True
AUGMENTATION_PROB = 0.3
NOISE_SCALE = 0.01
TIME_WARPING_PARAM = 0.2
```

## 📊 Kết Quả

Model được đánh giá bằng các metrics:

- **Loss**: Cross-entropy loss
- **Accuracy**: Token-level accuracy (bỏ qua padding)
- **BLEU Score**: Đánh giá chất lượng translation

### Ví dụ kết quả:

```
Test Results:
  Loss: 0.8234
  Accuracy: 0.8567
  BLEU Score: 0.7123

Sample Predictions:
1. Target:     xin chào bạn
   Prediction: xin chào bạn

2. Target:     tôi yêu việt nam
   Prediction: tôi yêu việt nam

3. Target:     cảm ơn rất nhiều
   Prediction: cảm ơn rất nhiều
```

## 💡 Tips & Best Practices

### 1. Data Preparation:

- ✅ Đảm bảo landmarks được extract đúng bằng MediaPipe
- ✅ Chuẩn hóa dữ liệu (đã được tự động xử lý trong code)
- ✅ Kiểm tra không có file .npy corrupt
- ✅ Label phải ở dạng text tiếng Việt viết thường

### 2. Training:

- ✅ **Bắt đầu với learning rate nhỏ** (0.001) và giảm dần
- ✅ **Sử dụng GPU** để training nhanh hơn (Google Colab cung cấp free GPU)
- ✅ **Monitor validation loss** để phát hiện overfitting sớm
- ✅ **Save checkpoints thường xuyên** (đã tự động)
- ✅ **Teacher forcing ratio giảm dần** theo epochs (đã tự động)

### 3. Overfitting:

Nếu thấy overfitting (val_loss tăng mà train_loss giảm):
- Tăng `DROPOUT_RATE` (0.3 → 0.5)
- Tăng data augmentation `AUGMENTATION_PROB` (0.3 → 0.5)
- Giảm model size (số filters, LSTM units)
- Thu thập thêm dữ liệu training

### 4. Underfitting:

Nếu cả train và val loss đều cao:
- Tăng model capacity (số layers, hidden dims)
- Giảm dropout
- Tăng số epochs
- Kiểm tra learning rate (có thể quá cao hoặc quá thấp)

### 5. Cải Thiện Performance:

- 📈 **Thu thập thêm dữ liệu**: Càng nhiều càng tốt
- 🎯 **Balance dataset**: Đảm bảo các class có số lượng sample tương đương
- 🔧 **Hyperparameter tuning**: Thử các giá trị khác
- 🏗️ **Thử Transformer**: Nếu có đủ dữ liệu (>10k samples)
- 🎭 **Ensemble models**: Kết hợp nhiều models

### 6. Inference:

- ✅ Sử dụng `best_model.pt` thay vì `latest_checkpoint.pt`
- ✅ Set `max_length` phù hợp với độ dài trung bình của câu
- ✅ Pre-process video giống như training data
- ✅ Batch prediction nhanh hơn single prediction

## 🐛 Troubleshooting

### Lỗi CUDA Out of Memory:

```bash
# Giảm batch size
python train.py --batch_size 8

# Hoặc giảm max_sequence_length trong config.py
MAX_SEQUENCE_LENGTH = 100  # thay vì 150
```

### Lỗi file không tồn tại:

```bash
# Kiểm tra đường dẫn trong meta.csv
# Đảm bảo Google Drive đã mount
```

### Model không học (loss không giảm):

```bash
# Thử learning rate khác
python train.py --learning_rate 0.0001

# Hoặc kiểm tra dữ liệu có đúng không
```

### BLEU score thấp:

- Kiểm tra tokenization có đúng không
- Đảm bảo vocabulary đủ lớn
- Train thêm epochs
- Tăng model capacity

## 📝 Files Trong Project

```
sign2vn/
├── config.py              # Cấu hình toàn bộ project
├── data_loader.py         # Data loading và preprocessing
├── model.py              # Model architecture
├── trainer.py            # Training logic
├── train.py              # Main training script
├── inference.py          # Inference script
├── requirements.txt      # Dependencies
├── Sign2VN_Training.ipynb  # Colab notebook
└── README.md            # File này
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo issue trên GitHub.

---

**Happy Training! 🚀**
