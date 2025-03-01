import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import ops

import numpy as np

import models as M
import tools

import config as conf

# Prepare the data
num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


image = x_train[np.random.choice(range(x_train.shape[0]))]
resized_image = ops.image.resize(
    ops.convert_to_tensor([image]), size=(conf.vit_config["image"]["image_size"], conf.vit_config["image"]["image_size"])
)

patches = M.Patches(patch_size=conf.vit_config["image"]["patch_size"])(resized_image)
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

tools.display_patches(patches, conf.vit_config["image"]["patch_size"], 3)

vit_model = M.create_vit(
    input_shape, 
    num_classes,
    conf.vit_config["image"],
    conf.vit_config["model"]
)

vit_model.layers[1].adapt(x_train)
print(vit_model.summary())