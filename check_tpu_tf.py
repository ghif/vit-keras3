import tensorflow as tf

# Provide the TPU name explicitly
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")

# You can then proceed with initialization
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.TPUStrategy(resolver)

print("All devices: ", tf.config.list_logical_devices('TPU'))