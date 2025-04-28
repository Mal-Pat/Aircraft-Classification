import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

import config
from model import AircraftCNN as TheModel
from dataset import get_dataloader as the_dataloader

cnn_model = TheModel(num_classes=config.NUM_CLASSES)

def run_training_loop(
    model = cnn_model, 
    num_epochs = config.NUM_EPOCHS, 
    train_loader = the_dataloader(
        data_dir=config.TRAIN_DATA_DIR,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE_DATA,
        num_workers=config.NUM_WORKERS
    ), 
    loss_fn = nn.CrossEntropyLoss(),
    optimizer = optim.Adam(cnn_model.parameters(), lr=config.LEARNING_RATE), 
    device = config.DEVICE,
    checkpoint_dir = config.CHECKPOINT_DIR,
    checkpoint_file = config.CHECKPOINT_FILE):

    print(f"Starting training on {device} for {num_epochs} epochs...")
    model.to(device)

    total_batches = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for batch_idx, (inputs, labels) in enumerate(train_loader):

            # Move data to the device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            current_loss = loss.item()
            # Accumulate weighted loss
            running_loss += current_loss * inputs.size(0)

            # Calculate accuracy for the batch
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Print progress periodically at below frequency
            print_freq = 50
            if (batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == total_batches:
                print(f"  Batch [{batch_idx+1}/{total_batches}] - "
                      f"Loss: {current_loss:.4f} "
                      )

        # Epoch Statistics
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        print(f"Epoch {epoch+1} Summary - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    print("\nTraining finished.")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the model in checkpoint directory
    torch.save(model.state_dict(), checkpoint_file)
    print(f"Model saved successfully to {checkpoint_file}.")

