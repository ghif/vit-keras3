import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import json

import keras
from keras import ops

import numpy as np

import models as M
import tools

import config_base as conf

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

# tools.display_patches(patches, conf.vit_config["image"]["patch_size"], 3)

# Use mixed precision
keras.mixed_precision.set_global_policy("mixed_float16")

vit_model = M.create_vit(
    input_shape, 
    num_classes,
    conf.vit_config["image"],
    conf.vit_config["model"]
)

vit_model.layers[1].adapt(x_train)
print(vit_model.summary())

for i, layer in enumerate(vit_model.layers):
    print(f"[{i}] {layer.name} - {layer.dtype_policy}")

# Train the model
optimizer = keras.optimizers.Adam(
    learning_rate=conf.vit_config["training"]["learning_rate"],
    weight_decay=conf.vit_config["training"]["weight_decay"],
    global_clipnorm=1.0
)

vit_model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

# Checkpoint callback
checkpoint_filepath = "models/vit_base_cifar100.weights.h5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)

history = vit_model.fit(
    x=x_train,
    y=y_train,
    batch_size=conf.vit_config["training"]["batch_size"],
    epochs=conf.vit_config["training"]["num_epochs"],
    # validation_split=0.1,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint_callback],
)

_, accuracy, top_5_accuracy = vit_model.evaluate(x_test, y_test, batch_size=conf.vit_config["training"]["batch_size"])
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

# Store history
history_dict = history.history

with open("models/vit_base_cifar100_history.json", "w") as f:
    json.dump(history_dict, f)