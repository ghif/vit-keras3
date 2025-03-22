# Vision Transformer (ViT) with Keras 3

This repository implements a Vision Transformer (ViT) architecture for image classification on CIFAR datasets. The implementation uses Keras with TensorFlow backend.

## Overview

Vision Transformers represent a novel approach to computer vision tasks by applying the transformer architecture, originally designed for NLP tasks, to image data. This implementation:

- Processes images by splitting them into fixed-size patches
- Linearly embeds each patch
- Adds positional embeddings
- Feeds the resulting sequence of vectors to a standard Transformer encoder
- Uses an MLP head for classification

## Model Architecture
The Vision Transformer implemented in this repository consists of:
1. Data Augmentation and Resizing: Random flips, rotations, and zooms
2. Patch Extraction: Converts images into patches
3. Patch Encoding: Linear projection and positional embedding
4. Transformer Encoder: Multiple layers of self-attention and MLP blocks
5. MLP Head: Final classification layers

## Installation
```bash
# Clone the repository
git clone https://github.com/ghif/vit-keras3.git
cd vit-keras3
```

## Project Structure

- `config_*.json`: JSON configuration files for Vision Transformer hyperparameters
- `models_vit.py`: Implementation of the ViT model architecture
- `dataset.py`: Data loading and preprocessing utilities
- `train_cifar100.py`: Script to train the model on CIFAR-100 dataset
- `finetune_cifar100.py`: Script to finetune a pre-trained model on CIFAR-100 dataset
- `inference_cifar100.py`: Script for performing inference on CIFAR-100 images
- `tools.py`: Utility functions for visualization and analysis

## Usage

To train the Vision Transformer from scratch on the CIFAR-100 dataset:
```python
python train_cifar100.py --config config_vit_base_96_train.json
```

To finetune a pre-trained Vision Transformer on the CIFAR-100 dataset:
```python
python finetune_cifar100.py --config config_vit_base_224_finetune.json
```

You can modify the model configuration by editing the `config_*.json` files. Available configurations include:
- `config_vit_base_96_train.json`: Configuration for training a ViT model from scratch with image size 96.
- `config_vit_base_224_finetune.json`: Configuration for finetuning a ViT model pre-trained on ImageNet with image size 224.
- `config_vit_base_224_finetune_all.json`: Configuration for finetuning all layers of a ViT model pre-trained on ImageNet with image size 224.

Example `config_vit_base_96_train.yaml`:
```json
{
    "image_shape": [96, 96, 3],
    "patch_size": 16,
    "num_layers": 12,
    "num_heads": 12,
    "mlp_dim": 3072,
    "attention_dropout_rate": 0.1,
    "dropout_rate": 0.1,
    "num_classes": 100,
    "learning_rate": 0.0001,
    "weight_decay": 0.03,
    "batch_size": 128,
    "epochs": 200,
    "global_clipnorm": 1.0
}
```


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
The MIT License is a permissive license that allows for reuse, modification, and distribution 
of this code for both private and commercial purposes, provided that the original copyright 
notice and permission notice are included. This software is provided "as is", without warranty of any kind.