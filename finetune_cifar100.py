import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import json

import keras_hub
import keras
import tensorflow as tf

import config_vit_base_224_finetune as conf
import dataset


# Contants
AUTOTUNE = tf.data.AUTOTUNE
MODEL_PREFIX = "vit_base_224_finetuned_v2"
BASE_MODEL = "vit_base_patch16_224_imagenet"

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


# Prepare the data
train_dataset, test_dataset, dataset_info = dataset.prepare_cifar100(conf.BATCH_SIZE, conf.IMAGE_SHAPE)

# # Check training images
# images = next(iter(train_dataset.take(1)))[0]
# tools.plot_image_gallery(images[:9], num_cols=3, figsize=(9, 9))

orig_image_shape = dataset_info.features["image"].shape
num_classes = dataset_info.features["label"].num_classes

# Use mixed precision
keras.mixed_precision.set_global_policy("mixed_float16")

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

print(image_classifier.summary(expand_nested=True))

# Check layer dtype policies
for i, layer in enumerate(image_classifier.layers):
    print(f"[{i}] {layer.name} - {layer.dtype_policy}")

steps_per_epoch = dataset_info.splits["train"].num_examples // conf.BATCH_SIZE
print(f"Steps per epoch: {steps_per_epoch}")
lr_schedule = get_cosine_decay_schedule(
    start_lr=conf.LEARNING_RATE,
    num_epochs=conf.EPOCHS,
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
    epochs=conf.EPOCHS,
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