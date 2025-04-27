# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_CHANNELS, NUM_CLASSES # Import necessary configs

class AircraftCNN(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):
        super(AircraftCNN, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten the output for the fully connected layers
        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply Convolutional Blocks
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten
        x = self.flatten(x)

        # Apply Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x