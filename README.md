# Aircraft Classification - Image and Video Processing Project

## Description

Militaries all over the world are investing in computer vision for surveillance, intelligence and reconnaisance. A part of this includes identifying large military aircraft in Air Force bases from satellite imagery. Automated monitoring of airfields and military installations can aid in tracking deployments, identifying potential threats, and assessing air force readiness. This technology can also be applied in humanitarian and disaster response efforts. By identifying usable airfields and assessing infrastructure damage from the air, effective emergency operations can be executed with better logistical precision.

## Goal

To classify 8 major types of military aircraft used by the United States Air Force (the dataset contains only U.S. aircraft) from their satellite images.

## Dataset

Link - [Google Drive Folder containing Satellite Images of 8 Classes of Military Aircraft](https://drive.google.com/drive/folders/1abQQCeZ92jGfF2C1GjnJQW8Mw_Jk9cw2)

I have excluded the C-17 aircraft class from the classification.

## Model

The model is a standard CNN (Convolutional Neural Network) built sequentially. It is a relatively shallow CNN with only 3 convolutional layers.

### Input
- Input images are preprocessed to 128px times 128px with 3 colour channels (RGB)
- Input shape is (batch size, 3, 128, 128)

### Convolution Block 1

- `nn.Conv2d(3, 16, kernel_size=3, padding=1)`: Applies 16 different 3x3 filters to the input image. padding=1 ensures the output spatial dimensions (height and width) remain 128x128. Output shape: `(batch_size, 16, 128, 128)`.
- `F.relu()`: Applies the Rectified Linear Unit activation function element-wise. Doesn't change the shape.
- `nn.MaxPool2d(kernel_size=2, stride=2)`: Performs max pooling over 2x2 regions with a stride of 2. This downsamples the spatial dimensions by half. Output shape: `(batch_size, 16, 64, 64)`.

### Convolution Block 2

- `nn.Conv2d(16, 32, kernel_size=3, padding=1)`: Applies 32 different 3x3 filters to the output of the previous block. Input channels (16) match the output channels of the previous Conv layer. Output shape: `(batch_size, 32, 64, 64)`.
- `F.relu()`: Applies the Rectified Linear Unit activation function element-wise. Doesn't change the shape.
- `nn.MaxPool2d(kernel_size=2, stride=2)`: Downsamples again. Output shape: `(batch_size, 32, 32, 32)`.

### Convolution Block 3

- `nn.Conv2d(32, 64, kernel_size=3, padding=1)`: Applies 64 different 3x3 filters. Output shape: `(batch_size, 64, 32, 32)`.
- `F.relu()`: Applies the Rectified Linear Unit activation function element-wise. Doesn't change the shape.
- `nn.MaxPool2d(kernel_size=2, stride=2)`: Final downsampling. Output shape: `(batch_size, 64, 16, 16)`.

### Flattening

- `nn.Flatten()`: Reshapes the 3D feature map (64, 16, 16) from each image in the batch into a 1D vector. The size of this vector is 64 * 16 * 16 = 16384. Output shape: `(batch_size, 16384)`.

### Fully Connected Classifier Block

- `nn.Linear(16384, 512)`: A standard dense layer connecting the 16384 features to 512 hidden units. Output shape: `(batch_size, 512)`.
- `F.relu()`: Activation function applied to the hidden layer.
- `nn.Dropout(0.5)`: Applies dropout with a probability of 0.5 during training. This helps prevent overfitting by randomly setting half of the inputs to zero. It is disabled during evaluation or prediction.
- `nn.Linear(512, num_classes)`: The final output layer. It connects the 512 hidden units to num_classes (9 in this case) output units, each corresponding to one of the aircraft types or bare land. Output shape: `(batch_size, 9)`.
