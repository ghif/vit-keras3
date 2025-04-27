import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import json
import numpy as np

import tensorflow as tf
import keras
import dataset

# Constants
BATCH_SIZE = 128
LEARNING_RATE = 1e-6
# WEIGHT_DECAY = 1e-4
GLOBAL_CLIPNORM = 1.0
EPOCHS = 2
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


loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn_test = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction="sum"
)

accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
top_5_accuracy_fn = keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")

def evaluate_model(model, tf_dataset):
    loss = 0.
    accuracy = 0.
    n_correct = 0

    top_5_accuracy = 0.
    n_top_5_correct = 0

    n_samples = 0
    
    for i, batch in enumerate(tf_dataset):
        image, label = batch
        pred = model.predict(image, verbose=None)

        # # Check for NaN or Inf in predictions
        # if np.isnan(pred).any() or np.isinf(pred).any():
        #     print("!!! NaN or Inf detected in predictions !!!")
        #     break
        bs = image.shape[0]
        # print(f"Batch-[{i+1}]: bs = {bs}")
        
        # Check loss
        loss_val = loss_fn_test(label, pred)
        loss += loss_val.numpy()
        
        # Check accuracy
        acc_val = accuracy_fn(label, pred)
        count_correct = acc_val.numpy() * bs
        n_correct += count_correct

        
        top_5_acc_val = top_5_accuracy_fn(label, pred)
        count_top_5_correct = top_5_acc_val.numpy() * bs
        n_top_5_correct += count_top_5_correct
        
        n_samples += bs
    
    loss /= n_samples
    accuracy = n_correct / n_samples
    top_5_accuracy = n_top_5_correct / n_samples
    
    return (loss, accuracy, top_5_accuracy)

class EvaluationCallback(keras.callbacks.Callback):
    def __init__(self, train_dataset, test_dataset):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
    
    def on_epoch_end(self, epoch, logs=None):
        print("Epoch {}:".format(epoch + 1))

        # Calculate evaluation manually
        (train_loss, train_acc, train_top_5_acc) = evaluate_model(self. model, self.train_dataset)
        (test_loss, test_acc, test_top_5_acc) = evaluate_model(self.model, self.test_dataset)
        
        print(f" > Train and test losses: ({train_loss:.4f}, {test_loss:.4f})")
        print(f" > Train and test accuracy: (top-1: {train_acc:.4f}, top-5: {train_top_5_acc:.4f}), (top-1: {test_acc:.4f}, top-5: {test_top_5_acc:.4f})")

class CheckWeightNaNs(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nChecking weights for NaNs at end of epoch {epoch+1}...")
        for layer in self.model.layers:
            for weight in layer.weights:
                if np.isnan(weight.numpy()).any() or np.isinf(weight.numpy()).any():
                    print(f"!!! NaN or Inf detected in weight: {weight.name} after epoch {epoch+1} !!!")
                    # Optionally stop training
                    # self.model.stop_training = True
                    return # Stop checking after first detection
        print("Weights seem OK.")

# Define full-connected networks with functional API
def mlp(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    logits = keras.layers.Dense(num_classes, dtype="float32")(x)
    return keras.Model(inputs=inputs, outputs=logits)


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
    epochs=EPOCHS,
    validation_data=test_dataset,
    # callbacks=[checkpoint_callback, EvaluationCallback(train_dataset, test_dataset)],
    callbacks=[checkpoint_callback],
)

print(f"== Evaluate model ==")
loss, accuracy, top_5_accuracy = evaluate_model(model, train_dataset)
# loss, accuracy, top_5_accuracy = model.evaluate(train_dataset)
print(f"Train loss: {loss}")
print(f"Train accuracy: {round(accuracy * 100, 2)}%")
print(f"Train top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

loss, accuracy, top_5_accuracy = evaluate_model(model, test_dataset)
# loss, accuracy, top_5_accuracy = model.evaluate(test_dataset)
print(f"Test loss: {loss}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

# Store history
history_dict = history.history

with open(f"models/{MODEL_PREFIX}_cifar100_history.json", "w") as f:
    json.dump(history_dict, f)