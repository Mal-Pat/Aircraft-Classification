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

# Dataset to load images
class AircraftDataset(Dataset):

    def __init__(self, data_dir, class_names=config.CLASS_NAMES, transform=data_transforms):
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
                print(f"Class directory not found: {class_dir}")
                continue

            for img_name in os.listdir(class_dir):
                # Check if files are images
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        if not self.image_paths:
            raise ValueError(f"No images found in {self.data_dir}.")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        # Open image in Pillow and ensure it is RGB
        image = Image.open(img_path).convert('RGB')

        label = self.labels[idx]

        # Apply transform to image
        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloader(data_dir=config.TRAIN_DATA_DIR,
                   batch_size=config.BATCH_SIZE,
                   shuffle=config.SHUFFLE_DATA,
                   num_workers=config.NUM_WORKERS,
                   transform=data_transforms,
                   class_names=config.CLASS_NAMES):

    dataset = AircraftDataset(data_dir=data_dir, class_names=class_names, transform=transform)

    # Check if dataset is empty before creating DataLoader
    if len(dataset) == 0:
        print(f"Dataset created from {data_dir} is empty.")
        return None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == torch.device("cuda") else False
    )
    return dataloader