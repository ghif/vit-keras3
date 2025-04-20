import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import jax

devices = jax.devices("tpu")
print(f"devices: {devices}")

# Define a 2x4 device mesh with data and model parallel axes
mesh = keras.distribution.DeviceMesh(
    shape=(2, 2), axis_names=["data", "model"], devices=devices
)

# A 2D layout, which describes how a tensor is distributed across the mesh
layout_2d = keras.distribution.TensorLayout(axes=("model", "data"), device_mesh=mesh)

# A 4D layout, which could be used for data parallel of an image input
replicated_layout_4d = keras.distribution.TensorLayout(
    axes=("data", None, None, None), device_mesh=mesh
)