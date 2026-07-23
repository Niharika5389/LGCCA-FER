import os
import torch

# Store the path to the dataset.
# If the code is running on Kaggle, use the Kaggle dataset path.
# Otherwise, use the local dataset path.

KAGGLE_DATASET_PATH = "/kaggle/input/fer2013"

LOCAL_DATASET_PATH = "data/FER2013"

if os.path.exists(KAGGLE_DATASET_PATH):
    DATASET_PATH = KAGGLE_DATASET_PATH
else:
    DATASET_PATH = LOCAL_DATASET_PATH


# All images will be resized to this size.
# EfficientNet-B0 expects 224 × 224 images.
IMAGE_SIZE = 224

# We have 7 facial emotion classes.
NUM_CLASSES = 7

# Number of images processed together before updating the model.
# Larger batch sizes train faster but require more GPU memory.
BATCH_SIZE = 32

# Number of times the entire dataset is passed through the model.
NUM_EPOCHS = 30

# Controls how much the model learns after every update.
# A value that is too large may overshoot the optimum.
# A value that is too small makes training very slow.
LEARNING_RATE = 1e-3

# Helps reduce overfitting by discouraging very large weights.
WEIGHT_DECAY = 1e-4

# Use the GPU if available, otherwise use the CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix the random seed so that results are reproducible.
SEED = 42