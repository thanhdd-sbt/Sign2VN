# 🚨 ACTION: 973 Videos - Quá Ít!

## 📊 Tình Huống

Bạn có:
- Total videos: 7558
- Matched: 7200
- **Processed: 973 only** ⚠️
- Skipped: 6585

**Vấn đề:** 973 entries quá ít! Nên có ~6800-7200 entries.

---

## 🔍 BƯỚC 1: Debug (1 phút)

Copy code này vào Colab:

```python
# Upload debug_labeling.py trước, sau đó:
%cd /content/Sign2VN
!python debug_labeling.py
```

**Output sẽ cho biết:**
1. ✅ Có bao nhiêu .npy files trong landmarks/
2. ✅ Có bao nhiêu entries trong meta.csv
3. ✅ Failed videos (nếu có)
4. ✅ Lý do chỉ có 973 entries

---

## 🎯 BƯỚC 2: Xác Định Nguyên Nhân

### **Scenario A: Có nhiều .npy files nhưng meta.csv ít**

**Nguyên nhân:** meta.csv bị corrupt/truncated

**Fix:**
```python
# Upload rebuild_meta.py, sau đó:
!python rebuild_meta.py
```

Script sẽ:
- ✅ Backup meta.csv cũ
- ✅ Scan tất cả .npy files
- ✅ Tạo meta.csv mới hoàn chỉnh

---

### **Scenario B: Có ít .npy files (chỉ ~973)**

**Nguyên nhân:** Extraction failed cho 6200+ videos

**Check failed_videos.txt:**
```python
!cat /content/drive/MyDrive/Sign2VN/work/failed_videos.txt | head -50
```

**Possible reasons:**
- Videos corrupt
- MediaPipe không detect được landmarks
- Videos quá ngắn (< 5 frames)

**Fix:** Re-run pipeline với settings mềm hơn:

```python
# Sửa trong labeling_config.py
MIN_FRAMES_REQUIRED = 3  # Thay vì 5
MAX_FRAMES_PER_VIDEO = None  # Không giới hạn

# Run lại
from data_labeling_pipeline import DataLabelingPipeline
pipeline = DataLabelingPipeline()
pipeline.run(resume=True)
```

---

### **Scenario C: Pipeline chạy trước đó đã process 6585 videos**

**Verify:**
```python
# Check số .npy files
!ls /content/drive/MyDrive/Sign2VN/work/landmarks/*.npy | wc -l
```

**Nếu output là ~6585-7000:**
- ✅ Pipeline đã chạy thành công trước đó!
- ✅ meta.csv có thể bị truncate
- ✅ Run rebuild_meta.py để fix

**Nếu output là ~973:**
- ⚠️ Most videos failed extraction
- ⚠️ Check failed_videos.txt
- ⚠️ Re-run với relaxed settings

---

## ✅ BƯỚC 3: Khắc Phục

### Option 1: Rebuild Meta.csv (Nếu có nhiều .npy)

```python
!python rebuild_meta.py
```

**Expected output:**
```
Found 6800 .npy files
✓ Created new meta.csv with 6800 entries
```

---

### Option 2: Re-run Pipeline (Nếu failed nhiều)

```python
# 1. Backup current data
!cp /content/drive/MyDrive/Sign2VN/work/meta.csv \
    /content/drive/MyDrive/Sign2VN/work/meta.csv.backup

# 2. Delete to start fresh
!rm /content/drive/MyDrive/Sign2VN/work/meta.csv

# 3. Adjust settings (optional)
import labeling_config as config
config.MIN_FRAMES_REQUIRED = 3
config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.3  # Lower threshold

# 4. Run pipeline
from data_labeling_pipeline import DataLabelingPipeline
pipeline = DataLabelingPipeline()
pipeline.run(resume=False)
```

---

### Option 3: Proceed với 973 Videos (Quick Test)

Nếu chỉ muốn test nhanh:

```bash
# Training với 973 videos để test
python train.py --num_epochs 10 --batch_size 8 --test

# Nếu training work → process thêm data sau
```

---

## 📊 Check Kết Quả

Sau khi fix:

```python
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/Sign2VN/work/meta.csv')

print(f"Total entries: {len(df)}")
print(f"Unique labels: {df['label_vi'].nunique()}")
print(f"Total frames: {df['num_frames'].sum():,}")

# Should see:
# Total entries: 6800+
# Unique labels: 1500+
# Total frames: 300,000+
```

---

## 💡 Quick Decision Tree

```
Có bao nhiêu .npy files?
├─ ~6800 files → Rebuild meta.csv (Option 1)
├─ ~973 files → Re-run pipeline (Option 2)
└─ Muốn test nhanh → Training với 973 (Option 3)
```

---

## 🚀 Recommended Action

**Chạy debug script trước:**
```python
!python debug_labeling.py
```

**Sau đó dựa vào output để quyết định Option 1, 2, hay 3.**

---

## 📞 Need Help?

Share kết quả của debug script:
```python
!python debug_labeling.py > /tmp/debug_output.txt
!cat /tmp/debug_output.txt
```

Copy toàn bộ output để tôi xem!

---

**Time to Fix:** 2-5 phút (với Option 1) hoặc 3 giờ (với Option 2)  
**Probability:** Option 1 (meta.csv corrupt) - 80%