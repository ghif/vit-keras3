import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import keras_hub

import dataset

import json
import argparse

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config_vit_base_224_finetune.json")
args = parser.parse_args()

# Load config json
with open(args.config) as f:
    conf = json.load(f)

IMAGE_SHAPE = tuple(conf["image_shape"])
LEARNING_RATE = conf["learning_rate"]
WEIGHT_DECAY = conf["weight_decay"]
BATCH_SIZE = conf["batch_size"]
EPOCHS = conf["epochs"]

def get_cosine_decay_schedule(
    start_lr, num_epochs, steps_per_epoch
):
    # calculate total steps
    total_steps = num_epochs * steps_per_epoch
    
    # Decay learning rate
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=start_lr,
        decay_steps=total_steps,
    )
    return lr_schedule

# Constants
BASE_MODEL = "vit_base_patch16_224_imagenet"


# Prepare data
train_dataset, test_dataset, dataset_info = dataset.prepare_cifar100(
    batch_size=BATCH_SIZE, target_image_shape=IMAGE_SHAPE
)


num_classes = dataset_info.features["label"].num_classes

# Use mixed precision
keras.mixed_precision.set_global_policy("mixed_float16")

preprocessor = keras_hub.models.ViTImageClassifierPreprocessor.from_preset(
    BASE_MODEL
)

backbone = keras_hub.models.Backbone.from_preset(BASE_MODEL)

image_classifier = keras_hub.models.ViTImageClassifier(
    backbone=backbone,
    num_classes=num_classes,
    preprocessor=preprocessor,
)
# Set DType Policy float32 for last layer
last_layer = image_classifier.layers[-1]
last_layer.dtype_policy = keras.mixed_precision.Policy("float32")

steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
print(f"Steps per epoch: {steps_per_epoch}")
lr_schedule = get_cosine_decay_schedule(
    start_lr=LEARNING_RATE,
    num_epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch
)
# Finetune the classifier with SGD optimizer
optimizer = keras.optimizers.SGD(
    learning_rate=lr_schedule, 
    momentum=0.9,
    global_clipnorm=1.0

)

image_classifier.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

print(image_classifier.summary(expand_nested=True))

# Load trained weights
image_classifier.load_weights("models/vit_base_224_finetuned_cifar100.weights.h5")

loss, accuracy, top_5_accuracy = image_classifier.evaluate(train_dataset)
print(f"Train loss: {loss}")
print(f"Train accuracy: {round(accuracy * 100, 2)}%")
print(f"Train top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

loss, accuracy, top_5_accuracy = image_classifier.evaluate(test_dataset)
print(f"Test loss: {loss}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")