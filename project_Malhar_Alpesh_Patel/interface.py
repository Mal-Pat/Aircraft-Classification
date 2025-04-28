import config

from model import AircraftCNN as TheModel

from train import run_training_loop as the_trainer

from predict import make_prediction as the_predictor

from dataset import AircraftDataset as TheDataset

from dataset import get_dataloader as the_dataloader

# This is used to transfer the data but isn't present in the instructions
from dataset import data_transforms as the_datatransform

# These were present in the original Final Project instructions, but not the recent one
# So I still added them in just in case
from config import NUM_EPOCHS as total_epochs
from config import BATCH_SIZE as the_batch_size