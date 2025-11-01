# 📊 Data Labeling Pipeline - Sign2VN

Pipeline tự động để trích xuất landmarks và tạo labels từ videos ngôn ngữ ký hiệu.

## 🎯 Mục Đích

Pipeline này sẽ:
1. ✅ Scan videos từ shared Google Drive folders
2. ✅ Match videos với dictionary để lấy labels tự động
3. ✅ Extract landmarks bằng MediaPipe
4. ✅ Tạo file `meta.csv` với labels hoàn chỉnh
5. ✅ Lưu landmarks thành file `.npy`

## 📁 Input Data Structure

### Shared Folders (trên Google Drive)
```
users/thanhnv/data/
├── videos/
│   ├── D0001B_địa_chỉ.mp4
│   ├── D0001N_địa_chỉ.mp4
│   ├── D0002_tỉnh.mp4
│   └── ...
├── videos_nnkh/
│   └── ...
└── videos/dictionary.json
```

### Dictionary Format
```json
[
  {
    "word": "địa chỉ",
    "_word": "dia chi",
    "description": "Những thông tin cụ thể về chỗ ở...",
    "tl": "Danh từ",
    "type": 0,
    "_id": "D0001B",
    "i": false,
    "local_video": "data/videos/D0001B_địa_chỉ.mp4",
    "video_url": "https://qipedc.moet.gov.vn/videos/D0001B.mp4?autoplay=true"
  }
]
```

## 📦 Output Structure

```
MyDrive/Sign2VN/
├── work/
│   ├── landmarks/              # Landmarks files (.npy)
│   │   ├── D0001B_địa_chỉ.npy
│   │   ├── D0002_tỉnh.npy
│   │   └── ...
│   ├── meta.csv               # Labels và metadata
│   ├── failed_videos.txt      # Videos thất bại
│   └── extraction_stats.json  # Statistics
```

### Meta.csv Format
```csv
npy,label,label_vi,orig_name,signer,description,type,num_frames,video_path
/path/to/D0001B_địa_chỉ.npy,D0001B,địa chỉ,D0001B_địa_chỉ.mp4,B,Những thông tin...,Danh từ,45,users/thanhnv/data/videos/D0001B_địa_chỉ.mp4
```

## 🚀 Quick Start (Google Colab)

### Option 1: Sử Dụng Notebook

1. Upload `Data_Labeling.ipynb` lên Google Colab
2. Chạy từng cell theo thứ tự
3. Monitor progress
4. Download kết quả

### Option 2: Command Line

```bash
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Setup
%cd /content/sign2vn
!python data_labeling_pipeline.py
```

## 📝 Chi Tiết Các Bước

### Bước 1: Setup Environment

```bash
# Cài đặt dependencies
pip install numpy==1.26.4
pip install opencv-python==4.8.1.78
pip install mediapipe==0.10.9
pip install pandas tqdm

# RESTART RUNTIME sau khi cài xong!
```

### Bước 2: Upload Code Files

Upload các files vào `/content/sign2vn/`:
- `labeling_config.py`
- `dictionary_manager.py`
- `video_scanner.py`
- `landmark_extractor.py`
- `data_labeling_pipeline.py`

### Bước 3: Cấu Hình Paths

Kiểm tra `labeling_config.py`:

```python
SHARED_FOLDERS = [
    "users/thanhnv/data/videos",
    "users/thanhnv/data/videos_nnkh",
]

DICTIONARY_PATH = "users/thanhnv/data/dictionary.json"

OUTPUT_ROOT = "/content/drive/MyDrive/Sign2VN"
```

### Bước 4: Test Components

```python
# Test Dictionary
from dictionary_manager import DictionaryManager
dm = DictionaryManager(config.DICTIONARY_PATH)
dm.load_dictionary()
dm.print_statistics()

# Test Video Scanner
from video_scanner import VideoScanner
scanner = VideoScanner(config.SHARED_FOLDERS)
videos = scanner.scan_videos()

# Test Extractor
from landmark_extractor import LandmarkExtractor
extractor = LandmarkExtractor()
landmarks = extractor.extract_from_video(videos[0]['full_path'])
```

### Bước 5: Chạy Pipeline

```python
from data_labeling_pipeline import DataLabelingPipeline

pipeline = DataLabelingPipeline()
pipeline.run(resume=True)
```

**Thời gian:** 1-3 giờ tùy số lượng videos

## ⚙️ Configuration

### labeling_config.py

Các settings quan trọng:

```python
# MediaPipe settings
MEDIAPIPE_MODEL_COMPLEXITY = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5

# Quality checks
MIN_FRAMES_REQUIRED = 5
MAX_FRAMES_PER_VIDEO = 300  # None = không giới hạn

# Processing
SKIP_EXISTING = True  # Skip videos đã xử lý
BATCH_SIZE = 100     # Save checkpoint mỗi 100 videos
```

## 🔍 Features

### 1. Auto Labeling
- Match videos với dictionary dựa trên filename
- Extract video ID (e.g., D0001B)
- Identify signer (B/N/T)

### 2. Checkpoint & Resume
- Tự động save checkpoint mỗi 100 videos
- Resume từ checkpoint nếu bị ngắt
- Skip videos đã xử lý

### 3. Quality Checks
- Validate video có đọc được không
- Check số frames tối thiểu
- Validate landmarks quality
- Log failed videos

### 4. Statistics
- Video distribution
- Label distribution
- Signer distribution
- Frame statistics

## 📊 Output Files

### 1. meta.csv
```python
import pandas as pd
df = pd.read_csv('Sign2VN/work/meta.csv')
print(df.head())
```

### 2. Landmarks (.npy)
```python
import numpy as np
landmarks = np.load('Sign2VN/work/landmarks/D0001B_địa_chỉ.npy')
print(f"Shape: {landmarks.shape}")  # (num_frames, 1629)
```

### 3. Statistics (JSON)
```python
import json
with open('Sign2VN/work/extraction_stats.json', 'r') as f:
    stats = json.load(f)
print(json.dumps(stats, indent=2))
```

### 4. Failed Videos Log
```
FAILED VIDEOS LOG
================================================================================

1. /path/to/video1.mp4
   Reason: Cannot open video

2. /path/to/video2.mp4
   Reason: Only 2 valid frames (min: 5)
```

## 🔧 Troubleshooting

### Video không tìm thấy
```python
# Kiểm tra paths
import os
print(os.path.exists('/content/drive/MyDrive/users/thanhnv/data/videos'))
```

### Dictionary không match
```python
# Test matching
dm = DictionaryManager(config.DICTIONARY_PATH)
dm.load_dictionary()

label = dm.get_label_for_video('D0001B_địa_chỉ.mp4')
print(f"Label: {label}")
```

### Landmarks extraction thất bại
```python
# Check MediaPipe
import mediapipe as mp
print(f"MediaPipe version: {mp.__version__}")

# Test với 1 video
extractor = LandmarkExtractor()
landmarks = extractor.extract_from_video('/path/to/video.mp4')
```

### Out of memory
```python
# Giảm MAX_FRAMES_PER_VIDEO trong config
MAX_FRAMES_PER_VIDEO = 150  # thay vì 300
```

## 💡 Tips

### 1. Batch Processing
- Pipeline tự động save checkpoint mỗi 100 videos
- Có thể dừng và resume bất cứ lúc nào
- Chạy ban đêm để tận dụng free GPU

### 2. Quality Control
```python
# Kiểm tra distribution
df = pd.read_csv('meta.csv')
print(df['label_vi'].value_counts())
print(df['num_frames'].describe())

# Loại bỏ outliers
df = df[df['num_frames'] >= 10]
df = df[df['num_frames'] <= 200]
```

### 3. Data Augmentation
- Pipeline chỉ extract raw landmarks
- Data augmentation được xử lý trong training
- Xem `data_loader.py` trong training code

## 📈 Expected Results

Với ~7500 videos:

```
Dictionary: 2595 unique words
Videos found: 7558
Matched: ~7200 (95%)
Successfully extracted: ~6800 (90%)

Average frames per video: 40-60
Total frames: ~400,000

Processing time: 2-3 hours on Colab GPU
```

## 🚦 Status Indicators

Pipeline progress:

```
[1/5] Loading dictionary... ✓
[2/5] Scanning videos... ✓
[3/5] Matching videos... ✓ 7200/7558 matched
[4/5] Loading existing data... ✓ 1000 entries
[5/5] Extracting landmarks...
  [1523/7200] Processing: D0523_...mp4
    ✓ Extracted 45 frames (2 failed)
  💾 Checkpoint saved: 1600 entries
```

## 📚 File Descriptions

### Core Files
- **labeling_config.py** - Configuration settings
- **dictionary_manager.py** - Load và match dictionary
- **video_scanner.py** - Scan videos từ folders
- **landmark_extractor.py** - Extract landmarks với MediaPipe
- **data_labeling_pipeline.py** - Main pipeline

### Notebook
- **Data_Labeling.ipynb** - Colab notebook với UI

## 🔄 Workflow

```
1. Load Dictionary
   ↓
2. Scan Videos
   ↓
3. Match Videos với Dictionary
   ↓
4. For each video:
   ├─ Extract landmarks
   ├─ Validate quality
   ├─ Save .npy file
   └─ Add to meta.csv
   ↓
5. Save results + statistics
```

## 📞 Support

Nếu gặp vấn đề:

1. Check `failed_videos.txt`
2. Review `extraction_stats.json`
3. Test với 1 video mẫu trước
4. Verify paths trong config

## ⏭️ Next Steps

Sau khi có data:

```bash
# 1. Verify meta.csv
head Sign2VN/work/meta.csv

# 2. Check statistics
cat Sign2VN/work/extraction_stats.json

# 3. Start training
python train.py --num_epochs 100 --batch_size 16 --test
```

---

**Version:** 1.0.0  
**Last Updated:** Nov 2025  
**Status:** ✅ Ready for Production
