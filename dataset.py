import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers

AUTOTUNE = tf.data.AUTOTUNE

random_flip = layers.RandomFlip("horizontal")
random_rotation = layers.RandomRotation(factor=0.1)

def preprocess_inputs(image, label):
    # image = tf.cast(image, tf.float32)
    image = tf.image.convert_image_dtype(image, tf.float32) # Converts to [0, 1]
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
   
def prepare(ds, batch_size, target_image_shape, st_type=-1, shuffle=False, augment=False):
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)

        # Resize the image to the target size
        image = tf.image.resize(image, [target_image_shape[0], target_image_shape[1]])
        
        if st_type >= 0:
            if st_type == 0:
                # Rescale the image to [0, 1]
                print(f"Rescale the image to [0, 1]")
                image = (image / 255.0)
            elif st_type == 1:
                # Per image standardization
                image = tf.image.per_image_standardization(image)
        
        return image, label
    
    # Resize and rescale all images
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set
    if augment:
        # Random horizontal flip
        ds = ds.map(
            lambda x, y: (random_flip(x), y),
            num_parallel_calls=AUTOTUNE,
        )

        # Random rotation
        ds = ds.map(
            lambda x, y: (random_rotation(x), y),
            num_parallel_calls=AUTOTUNE,
        )

    # Use buffered prefetching on all datasets
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds
  
def prepare_cifar100(batch_size, target_image_shape, st_type=0, augment=False):
    data, dataset_info = tfds.load("cifar100", with_info=True, as_supervised=True)
    train_dataset = data["train"]
    test_dataset = data["test"]

    train_dataset = prepare(
        train_dataset, 
        batch_size, 
        target_image_shape, 
        st_type=st_type, 
        shuffle=True, 
        augment=augment
    )
    test_dataset = prepare(
        test_dataset, 
        batch_size, 
        target_image_shape, 
        st_type=st_type, 
        shuffle=False, 
        augment=False
    )

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