import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import json
import math
import argparse

import keras_hub
import keras
from keras import ops
from keras.optimizers import schedules

import tensorflow as tf

import dataset

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config_vit_base_224_tpu_finetune.json")
args = parser.parse_args()

# Load config json
with open(args.config) as f:
    conf = json.load(f)


# Setup TPU configuration
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")

    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("Successfully initialized TPU.")
    print("All devices: ", tf.config.list_logical_devices('TPU'))
except Exception as e:
    print(f"Failed to initialize TPU: {e}")


def lr_warmup_cosine_decay(
    global_step,
    warmup_steps,
    hold=0,
    total_steps=0,
    target_lr=1e-3,
):
    # Cosine decay
    learning_rate = (
        0.5
        * target_lr
        * (
            1
            + ops.cos(
                math.pi
                * ops.convert_to_tensor(
                    global_step - warmup_steps - hold, dtype="float32"
                )
                / ops.convert_to_tensor(
                    total_steps - warmup_steps - hold, dtype="float32"
                )
            )
        )
    )

    warmup_lr = target_lr * (global_step / warmup_steps)

    if hold > 0:
        learning_rate = ops.where(
            global_step > warmup_steps + hold, learning_rate, target_lr
        )

    learning_rate = ops.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate

class WarmUpCosineDecay(schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_steps, hold, target_lr=1e-2):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def __call__(self, step):
        lr = lr_warmup_cosine_decay(
            global_step=step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            target_lr=self.target_lr,
            hold=self.hold,
        )
        return ops.where(step > self.total_steps, 0.0, lr)

# Contants
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SHAPE = tuple(conf["image_shape"])
LEARNING_RATE = conf["learning_rate"]
WEIGHT_DECAY = conf["weight_decay"]
BATCH_SIZE = conf["batch_size"]
EPOCHS = conf["epochs"]
AUGMENT = False
MIXED_PRECISION = True

if AUGMENT:
    MODEL_PREFIX = "vit_base_224_tpu_finetuned_aug"
else:
    MODEL_PREFIX = "vit_base_224_tpu_finetuned"

BASE_MODEL = "vit_base_patch16_224_imagenet"

# Prepare the data
train_dataset, test_dataset, dataset_info = dataset.prepare_cifar100(BATCH_SIZE, IMAGE_SHAPE, st_type=-1, augment=AUGMENT)
# prepare_cifar100(batch_size, target_image_shape, st_type=0, augment=False)

# # Check training images
# images = next(iter(train_dataset.take(1)))[0]
# tools.plot_image_gallery(images[:9], num_cols=3, figsize=(9, 9))

orig_image_shape = dataset_info.features["image"].shape
num_classes = dataset_info.features["label"].num_classes

if MIXED_PRECISION:
    # Use mixed precision
    keras.mixed_precision.set_global_policy("mixed_bfloat16")


steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
print(f"Steps per epoch: {steps_per_epoch}")

# calculate total steps
total_steps = EPOCHS * steps_per_epoch
print(f"Total steps: {total_steps}")

lr_schedule = WarmUpCosineDecay(
    target_lr=LEARNING_RATE,
    warmup_steps=int(0.1 * total_steps),
    total_steps=total_steps,
    hold=int(0.45 * total_steps)
)

with strategy.scope():

    backbone = keras_hub.models.Backbone.from_preset(BASE_MODEL)
    backbone.trainable = False

    preprocessor = keras_hub.models.ViTImageClassifierPreprocessor.from_preset(
        BASE_MODEL
    )

    image_classifier = keras_hub.models.ViTImageClassifier(
        backbone=backbone,
        num_classes=num_classes,
        preprocessor=preprocessor,
    )

    # Set DType Policy float32 for last layer
    last_layer = image_classifier.layers[-1]
    last_layer.dtype_policy = keras.mixed_precision.Policy("float32")

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

# Check layer dtype policies
for i, layer in enumerate(image_classifier.layers):
    print(f"[{i}] {layer.name} - {layer.dtype_policy}")

# Checkpoint callback
checkpoint_filepath = f"models/{MODEL_PREFIX}_cifar100.weights.h5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)

# Finetune the classifier
history = image_classifier.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=[checkpoint_callback],
)
loss, accuracy, top_5_accuracy = image_classifier.evaluate(train_dataset)
print(f"Train loss: {loss}")
print(f"Train accuracy: {round(accuracy * 100, 2)}%")
print(f"Train top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

loss, accuracy, top_5_accuracy = image_classifier.evaluate(test_dataset)
print(f"Test loss: {loss}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

# Store history
history_dict = history.history

with open(f"models/{MODEL_PREFIX}_cifar100_history.json", "w") as f:
    json.dump(history_dict, f)


# global_step = ops.arange(0, total_steps, dtype="int32")

# learning_rate = lr_schedule(global_step)
# print(learning_rate)

# # Plot learning rate
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6))
# plt.plot(learning_rate)
# plt.xlabel('Step')
# plt.ylabel('Learning Rate') 
# plt.title('Cosine Decay Learning Rate Schedule')
# # plt.show()
# plt.savefig("img/vit_base_224_finetuned_all_lr_schedule.png")