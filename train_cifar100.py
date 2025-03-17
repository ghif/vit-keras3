import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import json
import keras
import models_vit as M
import config_tiny as conf
import tensorflow_datasets as tfds

import config_vit_base_96 as conf

def get_dataset(batch_size, is_training=True):
  split = 'train' if is_training else 'test'
  dataset, info = tfds.load('cifar100', split=split, with_info=True, as_supervised=True, try_gcs=False)

  if is_training:
    dataset = dataset.shuffle(10000)

  dataset = dataset.batch(batch_size)
  return dataset, info

# Constants
MODEL_PREFIX = "vit_base_96"

# Prepare the data
num_classes = 100
input_shape = (32, 32, 3)

train_dataset, _ = get_dataset(conf.BATCH_SIZE, is_training=True)
test_dataset, _ = get_dataset(conf.BATCH_SIZE, is_training=False)

# Use mixed precision
keras.mixed_precision.set_global_policy("mixed_float16")

vit_model = M.vit_classifier(
    orig_image_shape=input_shape,
    image_shape=conf.IMAGE_SHAPE,
    patch_size=conf.PATCH_SIZE,
    num_layers=conf.NUM_LAYERS,
    num_heads=conf.NUM_HEADS,
    mlp_dim=conf.MLP_DIM,
    attention_dropout_rate=conf.ATTENTION_DROPOUT_RATE,
    dropout_rate=conf.DROPOUT_RATE,
    num_classes=conf.NUM_CLASSES
)

# vit_model.layers[1].adapt(x_train)
print(vit_model.summary())

for i, layer in enumerate(vit_model.layers):
    print(f"[{i}] {layer.name} - {layer.dtype_policy}")

# Train the model
optimizer = keras.optimizers.Adam(
    learning_rate=conf.LEARNING_RATE,
    weight_decay=conf.WEIGHT_DECAY,
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
checkpoint_filepath = f"models/{MODEL_PREFIX}_cifar100.weights.h5"

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)


history = vit_model.fit(
    train_dataset,
    batch_size=conf.BATCH_SIZE,
    epochs=conf.EPOCHS,
    validation_data=test_dataset,
    callbacks=[checkpoint_callback],
)
loss, accuracy, top_5_accuracy = vit_model.evaluate(train_dataset, batch_size=conf.BATCH_SIZE)
print(f"Train loss: {loss}")
print(f"Train accuracy: {round(accuracy * 100, 2)}%")
print(f"Train top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

loss, accuracy, top_5_accuracy = vit_model.evaluate(test_dataset, batch_size=conf.BATCH_SIZE)
print(f"Test loss: {loss}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

# Store history
history_dict = history.history

with open(f"models/{MODEL_PREFIX}_cifar100_history.json", "w") as f:
    json.dump(history_dict, f)