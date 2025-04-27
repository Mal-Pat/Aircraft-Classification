import torch
from PIL import Image
import os

import config
from model import AircraftCNN as TheModel
from dataset import data_transforms as the_datatransform

def load_model(model_path=config.CHECKPOINT_FILE, num_classes=config.NUM_CLASSES, device=config.DEVICE):

    model = TheModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model weights loaded successfully from {model_path}")

    model.to(device)
    # Set model to evaluation mode
    model.eval()
    return model

def preprocess_image(image_path, transform=the_datatransform):

    image = Image.open(image_path).convert('RGB')
    return transform(image)

def make_prediction(list_of_img_paths, class_names=config.CLASS_NAMES, device=config.DEVICE):
    
    model = load_model()
    predictions = []
    model.eval()

    # Disable gradient calculations for inference
    with torch.no_grad():
        for img_path in list_of_img_paths:
            image_tensor = preprocess_image(img_path)

            if image_tensor is None:
                predictions.append("Error: Could not process image")
                continue

            input_batch = image_tensor.unsqueeze(0).to(device)

            # Perform inference
            outputs = model(input_batch)

            # Get the index of the max log-probability
            _, predicted_idx = torch.max(outputs, 1)

            # Convert index to class name
            predicted_class = class_names[predicted_idx.item()]
            predictions.append(predicted_class)

    return predictions