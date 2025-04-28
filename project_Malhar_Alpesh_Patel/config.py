import torch

PROJECT_DIR = "project_malhar_alpesh_patel"
CHECKPOINT_DIR = f"{PROJECT_DIR}/checkpoints"
CHECKPOINT_FILE = f"{CHECKPOINT_DIR}/final_weights.pth"
DATA_EXAMPLE_DIR = f"{PROJECT_DIR}/data"

# Path to the main data
TRAIN_DATA_DIR = f"{PROJECT_DIR}/CompleteData"

RESIZE_HEIGHT = 128
RESIZE_WIDTH = 128
INPUT_CHANNELS = 3

# List of Class names which must match the folder names in TRAIN_DATA_DIR
CLASS_NAMES = [
    "B-1", "B-2", "B-52", "C-5", "C-130",
    "C-135", "E-3", "KC-10", "BareLand"
]
NUM_CLASSES = len(CLASS_NAMES)

# Normalization constants (using ImageNet stats as a common starting point)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 30
SHUFFLE_DATA = True

# My PC had cuda available on the Nvidia GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = 2

# Seed for reproducibility, in case needed
# It has not been used in the code currently
SEED = 100