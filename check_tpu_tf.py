import tensorflow as tf

try:
    # Inside the VM, 'local' or no argument often works
    # resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")

    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("Successfully initialized TPU.")
    print("All devices: ", tf.config.list_logical_devices('TPU'))
except Exception as e:
    print(f"Failed to initialize TPU: {e}")

# Provide the TPU name explicitly
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")

# # You can then proceed with initialization
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# # strategy = tf.distribute.TPUStrategy(resolver)

# print("All devices: ", tf.config.list_logical_devices('TPU'))