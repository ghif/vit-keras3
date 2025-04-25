import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import json
import numpy as np

import keras
import dataset



# Define loss function
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def evaluate_model(model, tf_dataset):
    losses = []
    for batch in tf_dataset:
        images, labels = batch
        predictions = model.predict(images, verbose=None)
        loss_batch = loss_fn(labels, predictions) # sum over batch size
        losses.append(loss_batch.numpy())

    loss = np.mean(losses)
    return loss

class EvaluationCallback(keras.callbacks.Callback):
    def __init__(self, train_dataset, test_dataset):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
    
    def on_epoch_end(self, epoch, logs=None):
        print("Epoch {}:".format(epoch + 1))

        train_loss = evaluate_model(self.model, self.train_dataset)
        test_loss = evaluate_model(self.model, self.test_dataset)

        print(f" > Train and test losses: ({train_loss:.4f}, {test_loss:.4f})")
        

# Define full-connected networks with functional API
def mlp(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # x = keras.layers.Normalization()(inputs)
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    logits = keras.layers.Dense(num_classes, dtype="float32")(x)
    return keras.Model(inputs=inputs, outputs=logits)

# Constants
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
GLOBAL_CLIPNORM = 1.0
EPOCHS = 100
MODEL_PREFIX = "mlp_noaug"


# Prepare the data
input_shape = (32, 32, 3)
train_dataset, test_dataset, dataset_info = dataset.prepare_cifar100_simple(BATCH_SIZE)

# Use mixed precision
keras.mixed_precision.set_global_policy("mixed_float16")

# Create the model
model = mlp(input_shape, dataset_info.features["label"].num_classes)

print(model.summary(expand_nested=True))

for i, layer in enumerate(model.layers):
    print(f"[{i}] {layer.name} - {layer.dtype_policy}")

# Train the model
# optimizer = keras.optimizers.SGD(
#     learning_rate=LEARNING_RATE,
#     momentum=0.9,
#     global_clipnorm=GLOBAL_CLIPNORM,
# )
optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    # weight_decay=WEIGHT_DECAY,
    global_clipnorm=GLOBAL_CLIPNORM,
)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
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
    callbacks=[checkpoint_callback, EvaluationCallback(train_dataset, test_dataset)],
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