import keras
from keras import layers
from keras import ops

def mlp(x, hidden_units, dropout_rate):
    """
    Create a multi-layer perceptron.
    Args:
        x (Tensor): Input tensor.
        hidden_units (list): List of integers with the number of units in each hidden layer.
        dropout_rate (float): Dropout rate.
    Returns:
        Tensor: Output tensor.
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def augment_and_resize(images, image_size):
    """
    Apply data augmentation and resize the input images.

    Args:
        images (Tensor): A batch of images with shape (batch_size, height, width, channels).
        image_size (int): The size of the output images.
    Returns:
        Tensor: The augmented and resized batch of images.
    """
    z = layers.Normalization()(images)
    z = layers.Resizing(image_size, image_size)(z)
    z = layers.RandomFlip("horizontal")(z)
    z = layers.RandomRotation(factor=0.02)(z)
    z = layers.RandomZoom(height_factor=0.2, width_factor=0.2)(z)
    return z


class Patches(layers.Layer):
    """Class to convert images into patches.
    This layer converts input images into a sequence of patches. The patches are created
    by dividing the image into equal-sized squares of size `patch_size x patch_size`.
    Args:
        patch_size (int): The size of each square patch (both width and height).
    Input shape:
        4D tensor with shape: (batch_size, height, width, channels)
    Output shape:
        3D tensor with shape: (batch_size, N, patch_size*patch_size*channels)
        where N = (height//patch_size) * (width//patch_size)
    Example:
        ```python
        patches = Patches(patch_size=16)
        x = tf.random.normal((2, 224, 224, 3))  # 2 images of 224x224x3
        patches_sequence = patches(x)  # Shape: (2, 196, 768)
        ```
    """
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(
            images,
            size=self.patch_size
        )
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
    
class PatchEncoder(layers.Layer):
    """A layer that combines linear projection and position embedding for image patches.
    This layer is part of Vision Transformer architecture that:
    1. Projects the flattened patches into a lower dimensional space
    2. Adds positional embeddings to retain spatial information
    Args:
        num_patches (int): Number of patches the image is divided into
        projection_dim (int): Dimension of the projection space
    Attributes:
        projection (layers.Dense): Linear projection layer
        position_embedding (layers.Embedding): Learnable position embeddings
    Call arguments:
        patch: A tensor of shape `(batch_size, num_patches, patch_dim)`
            where patch_dim is the flattened dimension of each image patch
    Returns:
        A tensor of shape `(batch_size, num_patches, projection_dim)`
        containing the projected patches with position embeddings added
    """
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    
    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

def create_vit(input_shape, num_classes, img_config, model_config):
    """
    Create a Vision Transformer model.
    Args:
        input_shape (tuple): Shape of the input tensor.
        num_classes (int): Number of output classes.
        img_config (dict): Dictionary containing parameters for image processing.
        model_config (dict): Dictionary containing parameters for the model.
    Returns:
        A Vision Transformer model (keras.Model).

    """
    image_size = img_config["image_size"]
    patch_size = img_config["patch_size"]
    num_patches = img_config["num_patches"]

    projection_dim = model_config["projection_dim"]
    transformer_layers = model_config["transformer_layers"]
    num_heads = model_config["num_heads"]
    transformer_units = model_config["transformer_units"]
    mlp_head_units = model_config["mlp_head_units"]

    inputs = keras.Input(shape=input_shape)

    # Augment data
    augmented = augment_and_resize(inputs, image_size)

    # Create patches
    patches = Patches(patch_size=patch_size)(augmented)

    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head self-attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=0.1
        )(x1, x1)

        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])
    # end for

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    # Add MLP
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    
    # Classify outputs
    logits = layers.Dense(num_classes)(features)

    # Create the Keras model
    model = keras.Model(inputs=inputs, outputs=logits)
    return model