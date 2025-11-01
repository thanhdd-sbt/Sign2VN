"""
Configuration for data labeling pipeline
"""

import os

# Google Drive paths
DRIVE_ROOT = "/content/drive/MyDrive"

# Shared folders
SHARED_FOLDERS = [
    "users/thanhnv/data/videos",
    "users/thanhnv/data/videos_nnkh",
]

# Dictionary path
DICTIONARY_PATH = "users/thanhnv/data/dictionary.json"

# Output paths
OUTPUT_ROOT = f"{DRIVE_ROOT}/Sign2VN"
LANDMARKS_OUTPUT_DIR = f"{OUTPUT_ROOT}/work/landmarks"
META_CSV_OUTPUT = f"{OUTPUT_ROOT}/work/meta.csv"
FAILED_VIDEOS_LOG = f"{OUTPUT_ROOT}/work/failed_videos.txt"
STATS_OUTPUT = f"{OUTPUT_ROOT}/work/extraction_stats.json"

# MediaPipe configuration
MEDIAPIPE_MODEL_COMPLEXITY = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# Landmarks configuration
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
NUM_FACE_LANDMARKS = 468
TOTAL_LANDMARKS = NUM_POSE_LANDMARKS + (NUM_HAND_LANDMARKS * 2) + NUM_FACE_LANDMARKS
LANDMARK_DIM = 3  # x, y, z

# Processing settings
SKIP_EXISTING = True  # Skip videos đã xử lý
MAX_FRAMES_PER_VIDEO = None  # Giới hạn số frames (None = không giới hạn)
MIN_FRAMES_REQUIRED = 1  # Số frames tối thiểu để video hợp lệ

# Video extensions to process
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

# Batch processing
BATCH_SIZE = 100  # Số videos xử lý trước khi save checkpoint

# Quality checks
CHECK_VIDEO_READABLE = True  # Kiểm tra video có đọc được không
CHECK_LANDMARKS_VALID = True  # Kiểm tra landmarks có hợp lệ không
MIN_CONFIDENCE_SCORE = 0.3  # Confidence tối thiểu cho landmarks

print(f"Output directory: {OUTPUT_ROOT}")
