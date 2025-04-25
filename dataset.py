import tensorflow as tf
import tensorflow_datasets as tfds
import keras

def check_data_numerics(image, label):
   tf.debugging.check_numerics(image, f"Data check failed for image")
   return image, label

def preprocess_inputs(image, label):
    image = tf.cast(image, tf.float32)
    return image, label

def prepare_cifar100_simple(batch_size, autotune=tf.data.AUTOTUNE):
    data, dataset_info = tfds.load("cifar100", with_info=True, as_supervised=True)
    train_dataset = data["train"]
    test_dataset = data["test"]

    train_dataset = train_dataset.shuffle(
      10 * batch_size, reshuffle_each_iteration=True
    ).map(preprocess_inputs, num_parallel_calls=autotune)
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = test_dataset.map(preprocess_inputs, num_parallel_calls=autotune)
    test_dataset = test_dataset.batch(batch_size)
    
    # test_dataset = test_dataset.prefetch(autotune)
    return train_dataset, test_dataset, dataset_info
    
def prepare_cifar100(batch_size, target_image_shape, autotune=tf.data.AUTOTUNE):
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
      10 * batch_size, reshuffle_each_iteration=True
    ).map(preprocess_inputs, num_parallel_calls=autotune)

    train_dataset = train_dataset.batch(batch_size)

    test_dataset = test_dataset.map(preprocess_inputs, num_parallel_calls=autotune)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset, dataset_info

def get_cifar100(batch_size, is_training=True, with_tpu=False):
    split = 'train' if is_training else 'test'
    dataset, info = tfds.load('cifar100', split=split, with_info=True, as_supervised=True, try_gcs=False)

    if is_training:
      dataset = dataset.shuffle(10000)

    if with_tpu:
       dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
      dataset = dataset.batch(batch_size)
    return dataset, info