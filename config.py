"""
Configuration file for Sign2VN project
"""

# Đường dẫn
DRIVE_PATH = "/content/drive/MyDrive/Sign2VN"
META_CSV_PATH = f"{DRIVE_PATH}/work/meta.csv"
LANDMARKS_DIR = f"{DRIVE_PATH}/work/landmarks"
CHECKPOINT_DIR = f"{DRIVE_PATH}/checkpoints"
LOGS_DIR = f"{DRIVE_PATH}/logs"

# MediaPipe landmarks
# Số lượng landmarks cho mỗi phần
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
NUM_FACE_LANDMARKS = 468

# Tổng số landmarks = pose + left_hand + right_hand + face
TOTAL_LANDMARKS = NUM_POSE_LANDMARKS + (NUM_HAND_LANDMARKS * 2) + NUM_FACE_LANDMARKS
LANDMARK_DIM = 3  # x, y, z coordinates

# Sequence settings
MAX_SEQUENCE_LENGTH = 150  # Độ dài tối đa của video (frames)
MIN_SEQUENCE_LENGTH = 10   # Độ dài tối thiểu

# Model hyperparameters
CNN_FILTERS = [64, 128, 256]
CNN_KERNEL_SIZE = 3
LSTM_UNITS = 512
LSTM_LAYERS = 2
DROPOUT_RATE = 0.3

# Seq2Seq settings
ENCODER_HIDDEN_DIM = 512
DECODER_HIDDEN_DIM = 512
ATTENTION_DIM = 256
EMBEDDING_DIM = 256

# Training settings
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7

# Vocabulary settings
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

# Device
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_PROB = 0.3
NOISE_SCALE = 0.01
TIME_WARPING_PARAM = 0.2

# Logging
LOG_INTERVAL = 10
SAVE_CHECKPOINT_EVERY = 5

print(f"Using device: {DEVICE}")
print(f"Total landmarks per frame: {TOTAL_LANDMARKS}")
