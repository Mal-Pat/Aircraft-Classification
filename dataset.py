# dataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config # Import configuration

# Define the transformations
# Use configurations for resize dimensions and normalization
data_transforms = transforms.Compose([
    transforms.Resize((config.RESIZE_HEIGHT, config.RESIZE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
])

class AircraftDataset(Dataset):
    """Custom Dataset for loading aircraft images."""

    def __init__(self, data_dir, class_names=config.CLASS_NAMES, transform=data_transforms):
        """
        Args:
            data_dir (string): Directory with all the images, structured with class subfolders.
            class_names (list): List of class names (folder names).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.image_paths = []
        self.labels = []

        # Check if data directory exists
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Populate image paths and labels
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue

            for img_name in os.listdir(class_dir):
                # Add basic image file check
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
                else:
                     print(f"Warning: Skipping non-image file: {os.path.join(class_dir, img_name)}")


        if not self.image_paths:
            raise ValueError(f"No images found in {self.data_dir}. Check directory structure and file types.")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        try:
            # Open image using Pillow
            image = Image.open(img_path).convert('RGB') # Ensure image is RGB
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder or skip? For now, re-raise maybe safer
            # Or you could return None and handle in DataLoader collate_fn
            raise e


        label = self.labels[idx]

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                 print(f"Error applying transform to image {img_path}: {e}")
                 raise e


        return image, label

def get_dataloader(data_dir=config.TRAIN_DATA_DIR,
                   batch_size=config.BATCH_SIZE,
                   shuffle=config.SHUFFLE_DATA,
                   num_workers=config.NUM_WORKERS,
                   transform=data_transforms,
                   class_names=config.CLASS_NAMES):
    """Creates a DataLoader for the AircraftDataset."""

    dataset = AircraftDataset(data_dir=data_dir, class_names=class_names, transform=transform)

    # Check if dataset is empty before creating DataLoader
    if len(dataset) == 0:
        print(f"Warning: Dataset created from {data_dir} is empty. DataLoader cannot be created.")
        return None # Or raise an error

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == torch.device("cuda") else False # Optimize CUDA transfers
    )
    return dataloader

# Example usage (optional)
if __name__ == '__main__':
    # Make sure the TRAIN_DATA_DIR in config.py points to your actual data
    print(f"Looking for data in: {config.TRAIN_DATA_DIR}")
    print(f"Expecting class folders: {config.CLASS_NAMES}")

    try:
        train_loader = get_dataloader()
        if train_loader:
            print(f"Successfully created DataLoader.")
            # Get one batch
            images, labels = next(iter(train_loader))
            print(f"Batch image shape: {images.shape}") # Should be [batch_size, 3, H, W]
            print(f"Batch labels shape: {labels.shape}") # Should be [batch_size]
            print(f"Example labels: {labels[:5]}")
        else:
            print("Failed to create DataLoader (Dataset might be empty).")

    except FileNotFoundError as e:
        print(e)
        print("Please ensure the 'TRAIN_DATA_DIR' in config.py is set correctly.")
    except ValueError as e:
        print(e)
        print("Please check the structure of your data directory and image files.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")