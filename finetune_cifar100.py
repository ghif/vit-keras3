import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_hub
import tensorflow_datasets as tfds
import keras
import tensorflow as tf

import config_vit_base_224_finetune as conf
import json

# Contants
AUTOTUNE = tf.data.AUTOTUNE
MODEL_PREFIX = "vit_base_224_finetuned"

def prepare_dataset(batch_size, target_image_shape):
    data, dataset_info = tfds.load("cifar100", with_info=True, as_supervised=True)
    train_dataset = data["train"]
    test_dataset = data["test"]
    
    resizing = keras.layers.Resizing(
      target_image_shape[0], target_image_shape[1], crop_to_aspect_ratio=True
    )

    def preprocess_inputs(image, label):
      image = tf.cast(image, tf.float32)
      return resizing(image), label
    
    train_dataset = train_dataset.shuffle(
      10 * conf.BATCH_SIZE, reshuffle_each_iteration=True
    ).map(preprocess_inputs, num_parallel_calls=AUTOTUNE)

    train_dataset = train_dataset.batch(batch_size)

    test_dataset = test_dataset.map(preprocess_inputs, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset, dataset_info


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
train_dataset, test_dataset, dataset_info = prepare_dataset(conf.BATCH_SIZE, conf.IMAGE_SHAPE)

# # Check training images
# images = next(iter(train_dataset.take(1)))[0]
# tools.plot_image_gallery(images[:9], num_cols=3, figsize=(9, 9))

orig_image_shape = dataset_info.features["image"].shape
num_classes = dataset_info.features["label"].num_classes

image_classifier = keras_hub.models.ImageClassifier.from_preset(
    "vit_base_patch16_224_imagenet",
    num_classes=num_classes
)
image_classifier.summary(expand_nested=True)

steps_per_epoch = dataset_info.splits["train"].num_examples // conf.BATCH_SIZE
print(f"Steps per epoch: {steps_per_epoch}")
lr_schedule = get_cosine_decay_schedule(
    start_lr=conf.LEARNING_RATE,
    num_epochs=conf.EPOCHS,
    steps_per_epoch=steps_per_epoch
)
# Finetune the classifier with SGD optimizer
optimizer = keras.optimizers.SGD(
    learning_rate=conf.LEARNING_RATE, 
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
loss, accuracy, top_5_accuracy = image_classifier.evaluate(train_dataset, batch_size=conf.BATCH_SIZE)
print(f"Train loss: {loss}")
print(f"Train accuracy: {round(accuracy * 100, 2)}%")
print(f"Train top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

loss, accuracy, top_5_accuracy = image_classifier.evaluate(test_dataset, batch_size=conf.BATCH_SIZE)
print(f"Test loss: {loss}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

# Store history
history_dict = history.history

with open(f"models/{MODEL_PREFIX}_cifar100_history.json", "w") as f:
    json.dump(history_dict, f)