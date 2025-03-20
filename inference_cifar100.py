import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import keras_hub

import dataset
import config_vit_base_224_finetune as conf

# Constants
BASE_MODEL = "vit_base_patch16_224_imagenet"


# Prepare data
train_dataset, test_dataset, dataset_info = dataset.prepare_cifar100(
    batch_size=conf.BATCH_SIZE, target_image_shape=conf.IMAGE_SHAPE
)


num_classes = dataset_info.features["label"].num_classes

# Use mixed precision
keras.mixed_precision.set_global_policy("mixed_float16")

backbone = keras_hub.models.Backbone.from_preset(BASE_MODEL)

preprocessor = keras_hub.models.ViTImageClassifierPreprocessor.from_preset(
    BASE_MODEL
)

image_classifier = keras_hub.models.ViTImageClassifier(
    backbone=backbone,
    num_classes=num_classes,
    preprocessor=preprocessor,
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