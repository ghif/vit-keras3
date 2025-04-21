import tensorflow as tf

# Provide the TPU name explicitly
tpu_name = "tpu-v4-base-us"
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)

# You can then proceed with initialization
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

print("All devices: ", tf.config.list_logical_devices('TPU'))