import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import jax

devices = jax.devices("tpu")
print(f"devices: {devices}")

# Define a 2x4 device mesh with data and model parallel axes
mesh = keras.distribution.DeviceMesh(
    shape=(2, 4), axis_name=["data", "model"], devices=devices
)