import os
import torch

# Base path for dataset
BASE_PATH = "/kaggle/input"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Audio & frame-related settings
FRAME_RATE = 86
INTERVAL_FRAMES = 1290        # 15 seconds
STRIDE_FRAMES = 946           # 11 seconds overlap
MEL_SIZE = (128, INTERVAL_FRAMES)
SSM_TARGET_SIZE = (64, 64)

# Training hyperparameters
BATCH_SIZE = 1
EPOCHS = 10
VAL_SPLIT = 0.15

# Checkpoint path
CHECKPOINT_PATH = "/kaggle/working/best_model.pt"
