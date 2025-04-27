import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import json
import argparse
import numpy as np

import keras
import tensorflow_datasets as tfds

import models_vit as M
import dataset

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config_vit_base_96_train.json")
args = parser.parse_args()

# Load config json
with open(args.config) as f:
    conf = json.load(f)

# Constants
MODEL_PREFIX = "vit_base_96_noaug"

IMAGE_SHAPE = tuple(conf["image_shape"])
PATCH_SIZE = conf["patch_size"]
NUM_LAYERS = conf["num_layers"]
NUM_HEADS = conf["num_heads"]
MLP_DIM = conf["mlp_dim"]
ATTENTION_DROPOUT_RATE = conf["attention_dropout_rate"]
DROPOUT_RATE = conf["dropout_rate"]
NUM_CLASSES = conf["num_classes"]
LEARNING_RATE = conf["learning_rate"]
WEIGHT_DECAY = conf["weight_decay"]
BATCH_SIZE = conf["batch_size"]
EPOCHS = conf["epochs"]
GLOBAL_CLIPNORM = conf["global_clipnorm"]

# Prepare the data
train_dataset, test_dataset, dataset_info = dataset.prepare_cifar100(BATCH_SIZE, IMAGE_SHAPE, st_type=0, augment=True)

# Use mixed precision
keras.mixed_precision.set_global_policy("mixed_float16")

vit_model = M.vit_classifier(
    image_shape=IMAGE_SHAPE,
    patch_size=PATCH_SIZE,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM,
    attention_dropout_rate=ATTENTION_DROPOUT_RATE,
    dropout_rate=DROPOUT_RATE,
    num_classes=NUM_CLASSES
)

# Print model
print(vit_model.summary(expand_nested=True))

for i, layer in enumerate(vit_model.layers):
    print(f"[{i}] {layer.name} - {layer.dtype_policy}")

# Train the model
optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    global_clipnorm=GLOBAL_CLIPNORM,
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
checkpoint_filepath = f"models/{MODEL_PREFIX}_cifar100.weights.h5"

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)

history = vit_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=[checkpoint_callback],
)

loss, accuracy, top_5_accuracy = vit_model.evaluate(train_dataset, batch_size=BATCH_SIZE)
print(f"Train loss: {loss}")
print(f"Train accuracy: {round(accuracy * 100, 2)}%")
print(f"Train top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

loss, accuracy, top_5_accuracy = vit_model.evaluate(test_dataset, batch_size=BATCH_SIZE)
print(f"Test loss: {loss}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

# Store history
history_dict = history.history

with open(f"models/{MODEL_PREFIX}_cifar100_history.json", "w") as f:
    json.dump(history_dict, f)