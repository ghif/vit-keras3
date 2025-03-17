import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_hub
import tensorflow_datasets as tfds
import keras

def get_dataset(batch_size, is_training=True):
  split = 'train' if is_training else 'test'
  dataset, info = tfds.load('cifar100', split=split, with_info=True, as_supervised=True, try_gcs=False)

  if is_training:
    dataset = dataset.shuffle(10000)

  dataset = dataset.batch(batch_size)
  return dataset, info

# Prepare the data
num_classes = 100
batch_size = 128
input_shape = (32, 32, 3)

train_dataset, _ = get_dataset(batch_size, is_training=True)
test_dataset, _ = get_dataset(batch_size, is_training=False)

image_classifier = keras_hub.models.ImageClassifier.from_preset(
    "vit_base_patch16_224_imagenet"
)
image_classifier.summary(expand_nested=True)
image_classifier.backbone.summary(expand_nested=True)
image_classifier.backbone.summary(expand_nested=True)
keras.utils.plot_model(image_classifier.backbone, "vit_base_patch16_224_imagenet.png", show_shapes=True)