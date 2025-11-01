# ⚡ Quick Start - Data Labeling

Hướng dẫn nhanh để bắt đầu labeling data trong 15 phút.

## 🎯 Mục Tiêu

Trích xuất landmarks và tạo labels tự động từ videos trong shared folders.

## ⏱️ Thời Gian

- **Setup:** 5 phút
- **Test:** 5 phút  
- **Run:** 2-3 giờ (tự động)

---

## 📋 Bước 1: Setup Colab (3 phút)

### 1.1 Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 1.2 Cài Packages
```python
import sys
!{sys.executable} -m pip uninstall -y opencv-python-headless
!{sys.executable} -m pip install -q numpy==1.26.4
!{sys.executable} -m pip install -q opencv-python==4.8.1.78
!{sys.executable} -m pip install -q mediapipe==0.10.9
!{sys.executable} -m pip install -q pandas tqdm
```

### 1.3 RESTART RUNTIME
**Runtime → Restart runtime** (Bắt buộc!)

### 1.4 Verify
```python
import cv2, mediapipe, numpy, pandas
print("✓ Ready!")
```

---

## 📦 Bước 2: Upload Code (2 phút)

Upload 5 files này vào `/content/sign2vn/`:
1. `labeling_config.py`
2. `dictionary_manager.py`
3. `video_scanner.py`
4. `landmark_extractor.py`
5. `data_labeling_pipeline.py`

```python
!mkdir -p /content/sign2vn
%cd /content/sign2vn
!ls -lh
```

---

## 🧪 Bước 3: Test (5 phút)

### 3.1 Test Dictionary
```python
import sys
sys.path.append('/content/sign2vn')

from dictionary_manager import DictionaryManager
import labeling_config as config

dm = DictionaryManager(config.DICTIONARY_PATH)
dm.load_dictionary()
dm.print_statistics()
```

**Expected output:**
```
✓ Loaded 2595 entries
✓ Built lookup tables:
  - 2595 video IDs
  - 850 unique words
```

### 3.2 Test Video Scanner
```python
from video_scanner import VideoScanner

scanner = VideoScanner(config.SHARED_FOLDERS)
videos = scanner.scan_videos()
scanner.print_statistics()
```

**Expected output:**
```
✓ Found 3500 videos
✓ Found 4058 videos
TOTAL VIDEOS FOUND: 7558
```

### 3.3 Test Extractor (1 video)
```python
from landmark_extractor import LandmarkExtractor

if videos:
    extractor = LandmarkExtractor()
    test_video = videos[0]['full_path']
    print(f"Testing: {videos[0]['filename']}")
    
    landmarks = extractor.extract_from_video(test_video)
    
    if landmarks is not None:
        print(f"✓ Success! Shape: {landmarks.shape}")
    else:
        print("✗ Failed")
```

**Expected output:**
```
Testing: D0001B_địa_chỉ.mp4
  ✓ Extracted 45 frames (0 failed)
✓ Success! Shape: (45, 1629)
```

---

## 🚀 Bước 4: Run Pipeline (2-3 giờ tự động)

```python
from data_labeling_pipeline import DataLabelingPipeline

pipeline = DataLabelingPipeline()
pipeline.run(resume=True)
```

**Progress output:**
```
================================================================================
SIGN LANGUAGE DATA LABELING PIPELINE
================================================================================
Start time: 2025-11-01 10:00:00

[Step 1/5] Loading dictionary...
✓ Loaded 2595 entries

[Step 2/5] Scanning videos...
TOTAL VIDEOS FOUND: 7558

[Step 3/5] Matching videos with dictionary...
Matched: 7200

[Step 4/5] Loading existing data...
✓ Loaded 0 previously processed videos

[Step 5/5] Extracting landmarks...
Processing 7200 videos...
================================================================================

[1/7200] Processing: D0001B_địa_chỉ.mp4
  ✓ Extracted 45 frames (0 failed)

[2/7200] Processing: D0001N_địa_chỉ.mp4
  ✓ Extracted 52 frames (1 failed)

...

  💾 Checkpoint saved: 100 entries
  💾 Checkpoint saved: 200 entries
  ...
```

---

## ✅ Bước 5: Verify Results (2 phút)

### 5.1 Check meta.csv
```python
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/Sign2VN/work/meta.csv')
print(f"Total entries: {len(df)}")
print(df.head())
```

### 5.2 Check landmarks
```python
import numpy as np

sample_npy = df.iloc[0]['npy']
landmarks = np.load(sample_npy)
print(f"Shape: {landmarks.shape}")
print(f"Frames: {len(landmarks)}")
```

### 5.3 Check statistics
```python
import json

with open('/content/drive/MyDrive/Sign2VN/work/extraction_stats.json', 'r') as f:
    stats = json.load(f)

print(json.dumps(stats, indent=2))
```

**Expected stats:**
```json
{
  "timestamp": "2025-11-01T13:00:00",
  "extraction_stats": {
    "total_videos": 7200,
    "successful": 6800,
    "failed": 400,
    "total_frames": 300000,
    "success_rate": "94.4%"
  }
}
```

---

## 🎉 Done!

Bây giờ bạn có:
- ✅ **meta.csv** với 6800+ labeled videos
- ✅ **landmarks/** với 6800+ .npy files
- ✅ **extraction_stats.json** với statistics
- ✅ **failed_videos.txt** với danh sách videos lỗi

---

## ⏭️ Next Step: Training

```bash
python train.py --num_epochs 100 --batch_size 16 --test
```

---

## 💡 Tips

### Dừng và Resume
```python
# Nếu bị ngắt, chỉ cần chạy lại:
pipeline.run(resume=True)

# Pipeline sẽ:
# 1. Load checkpoint
# 2. Skip videos đã xử lý
# 3. Tiếp tục từ chỗ dừng
```

### Skip Videos Đã Có
```python
# Trong labeling_config.py:
SKIP_EXISTING = True  # Skip videos đã xử lý
```

### Giới Hạn Frames
```python
# Trong labeling_config.py:
MAX_FRAMES_PER_VIDEO = 300  # Giới hạn frames
```

---

## 🔍 Troubleshooting

### Videos không tìm thấy
```python
# Check paths
import os
path = "/content/drive/MyDrive/users/thanhnv/data/videos"
print(f"Exists: {os.path.exists(path)}")
```

### Dictionary không match
```python
# Test matching
label = dm.get_label_for_video('D0001B_địa_chỉ.mp4')
print(f"Label: {label}")  # Should print: "địa chỉ"
```

### Extraction lỗi
```python
# Check MediaPipe
import mediapipe as mp
print(f"MediaPipe: {mp.__version__}")
```

---

**Total Time:** ~3 hours (mostly automated)  
**Difficulty:** ⭐⭐ Easy  
**Output:** Ready-to-train dataset
