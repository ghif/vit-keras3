import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import json

import tensorflow as tf
import keras
import dataset

# Define full-connected networks with functional API
def mlp(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(512, activation="relu")(x)
    logits = keras.layers.Dense(num_classes, dtype="float32")(x)
    return keras.Model(inputs=inputs, outputs=logits)

# Constants
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
# WEIGHT_DECAY = 1e-4
GLOBAL_CLIPNORM = 1.0
EPOCHS = 100
MODEL_PREFIX = "mlp_noaug"

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


# Prepare the data
input_shape = (32, 32, 3)
train_dataset, test_dataset, dataset_info = dataset.prepare_cifar100_simple(BATCH_SIZE)

# # Use mixed precision
# keras.mixed_precision.set_global_policy("mixed_float16")

with strategy.scope():
    # Create the model
    model = mlp(input_shape, dataset_info.features["label"].num_classes)

    print(model.summary(expand_nested=True))

    for i, layer in enumerate(model.layers):
        print(f"[{i}] {layer.name} - {layer.dtype_policy}")

    # Train the model
    optimizer = keras.optimizers.SGD(
        learning_rate=LEARNING_RATE,
        momentum=0.9,
        global_clipnorm=GLOBAL_CLIPNORM,
    )
    # optimizer = keras.optimizers.Adam(
    #     learning_rate=LEARNING_RATE,
    #     global_clipnorm=GLOBAL_CLIPNORM,
    # )

    model.compile(
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

history = model.fit(
    train_dataset,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_dataset,
    steps_per_epoch=dataset_info.splits["train"].num_examples // BATCH_SIZE,
    validation_steps=dataset_info.splits["validation"].num_examples // BATCH_SIZE,
    callbacks=[checkpoint_callback],
)

loss, accuracy, top_5_accuracy = model.evaluate(train_dataset, batch_size=BATCH_SIZE)
print(f"Train loss: {loss}")
print(f"Train accuracy: {round(accuracy * 100, 2)}%")
print(f"Train top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

loss, accuracy, top_5_accuracy = model.evaluate(test_dataset, batch_size=BATCH_SIZE)
print(f"Test loss: {loss}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

# Store history
history_dict = history.history

with open(f"models/{MODEL_PREFIX}_cifar100_history.json", "w") as f:
    json.dump(history_dict, f)