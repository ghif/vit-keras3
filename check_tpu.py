import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import jax

devices = jax.devices("gpu")
print(f"devices: {devices}")