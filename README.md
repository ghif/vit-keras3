# Vision Transformer (ViT) with Keras 3
This repository implements a Vision Transformer (ViT) architecture for image classification on CIFAR datasets. The implementation uses Keras with TensorFlow backend.

## Overview
Vision Transformers represent a novel approach to computer vision tasks by applying the transformer architecture, originally designed for NLP tasks, to image data. This implementation:
- Processes images by splitting them into fixed-size patches
- Linearly embeds each patch
- Adds positional embeddings
- Feeds the resulting sequence of vectors to a standard Transformer encoder
- Uses an MLP head for classification

## Installation
```
# Clone the repository
git clone https://github.com/ghif/vit-keras3.git
cd vit-keras3
```

## Project Structure
- config.py: Configuration dictionary for Vision Transformer hyperparameters
- models.py: Implementation of the ViT model architecture
- tools.py: Utility functions for visualization and analysis
- train_cifar100.py: Script to train the model on CIFAR-100 dataset

## Usage

To train the Vision Transformer on the CIFAR-100 dataset:

```
python train_cifar100.py
```

You can modify the model configuration in config.py to change hyperparameters such as:
- Image size
- Patch size
- Transformer architecture (depth, width, heads)
- MLP head configuration
- Training parameters (learning rate, batch size, etc.)

## Model Architecture
The Vision Transformer implemented in this repository consists of:
1. Data Augmentation and Resizing: Random flips, rotations, and zooms
2. Patch Extraction: Converts images into patches
3. Patch Encoding: Linear projection and positional embedding
4. Transformer Encoder: Multiple layers of self-attention and MLP blocks
5. MLP Head: Final classification layers

## Configuration
The model is configurable through the dictionary in `config_*.py`:
```
# Model architecture
IMAGE_SHAPE = (96, 96, 3)
PATCH_SIZE = 16
NUM_LAYERS = 12
NUM_HEADS = 12
MLP_DIM = 3072
ATTENTION_DROPOUT_RATE = 0.1
DROPOUT_RATE = 0.1
NUM_CLASSES = 100

# Training
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.03
BATCH_SIZE = 128
EPOCHS = 200
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
The MIT License is a permissive license that allows for reuse, modification, and distribution 
of this code for both private and commercial purposes, provided that the original copyright 
notice and permission notice are included. This software is provided "as is", without warranty of any kind.