import keras
from keras import layers
from keras import ops

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

def mlp_block(x, mlp_dim, dropout_rate):
    """
    Create an MLP block.

    Args:
        x (Tensor): Input tensor.
        mlp_dim (int): Hidden dimension of the MLP block.
        dropout_rate (float): Dropout rate for the block.
    Returns:
        Output tensor.
    """
    y = layers.Dense(mlp_dim, activation=keras.activations.gelu)(x)
    y = layers.Dropout(dropout_rate)(y)
    y = layers.Dense(x.shape[-1])(y)
    y = layers.Dropout(dropout_rate)(y)
    return y

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
    mlp_size = model_config["mlp_size"]
    encoder_dropout_rate = model_config["encoder_dropout_rate"]
    head_dropout_rate = model_config["head_dropout_rate"]

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
            dropout=encoder_dropout_rate
        )(x1, x1)

        # print(f"attention_output shape: {attention_output.shape}")

        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP
        # x3 = mlp(x3, hidden_units=[projection_dim], dropout_rate=0.1)
        x3 = mlp_block(x3, hidden_unit=mlp_size, dropout_rate=encoder_dropout_rate)

        # Skip connection 2
        # print(f"x2 shape: {x2.shape}")
        # print(f"x3 shape: {x3.shape}")
        encoded_patches = layers.Add()([x3, x2])
    # end for

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    features = layers.Dropout(head_dropout_rate)(representation)

    # # Add MLP
    # features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.1)
    
    # Classify outputs
    logits = layers.Dense(num_classes, dtype="float32")(features)

    # Create the Keras model
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def encoder1d_block(inputs, num_heads, hidden_dim, mlp_dim, attention_dropout_rate, dropout_rate):
    """
    Create an Encoder 1D block.
    Args:
        inputs (Tensor): Input tensor.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Hidden dimension of the feedforward network.
        mlp_dim (int): Hidden dimension of the MLP block.
        attention_dropout_rate (float): Dropout rate for the attention layer.
        dropout_rate (float): Dropout rate for the block.
    Returns:
        Output tensor.
    """
    # Layer normalization 1
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)

    key_dim = hidden_dim // num_heads

    # Multi Head Attention layer
    x = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=attention_dropout_rate
    )(x, x)

    # Dropout
    x = layers.Dropout(dropout_rate)(x)

    # Skip connection 1
    x = layers.Add()([x, inputs])

    # MLP block
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = mlp_block(y, mlp_dim, dropout_rate)
    y = layers.Add()([y, x])


    # Skip connection 2
    return y

def vit_encoder(inputs, num_layers, num_heads, hidden_dim, mlp_dim, attention_dropout_rate, dropout_rate):
    """
    Create a Vision Transformer encoder.
    Args:
        inputs (Tensor): Input tensor.
        num_layers (int): Number of encoder layers.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Hidden dimension of the feedforward network.
        mlp_dim (int): Hidden dimension of the MLP block.
        attention_dropout_rate (float): Dropout rate for the attention layer.
        dropout_rate (float): Dropout rate for the block.
    Returns:
        Output tensor.
    """
    x = layers.Dropout(dropout_rate)(inputs)
    for _ in range(num_layers):
        x = encoder1d_block(
            x, num_heads, hidden_dim, mlp_dim, attention_dropout_rate, dropout_rate
        )
    return x

def vit_backbone(orig_image_shape, image_shape, patch_size, num_layers, num_heads, mlp_dim, attention_dropout_rate, dropout_rate):
    num_patches = (image_shape[0] // patch_size) * (image_shape[1] // patch_size)
    hidden_dim = patch_size * patch_size * 3

    inputs = keras.Input(shape=orig_image_shape)
    augmented = augment_and_resize(inputs, image_shape[0])
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, hidden_dim)(patches)

    y = vit_encoder(
        encoded_patches, num_layers, num_heads, hidden_dim, mlp_dim, attention_dropout_rate, dropout_rate
    )
    return keras.Model(inputs=inputs, outputs=y)

def vit_classifier(orig_image_shape, image_shape, patch_size, num_layers, num_heads, mlp_dim, attention_dropout_rate, dropout_rate, num_classes):
    backbone = vit_backbone(
        orig_image_shape=orig_image_shape,
        image_shape=image_shape,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        attention_dropout_rate=attention_dropout_rate,
        dropout_rate=dropout_rate
    )
    inputs = backbone.inputs

    features = backbone(inputs)

    h = layers.GlobalAveragePooling1D()(features)

    h = layers.Dropout(dropout_rate)(h)
    logits = layers.Dense(num_classes, dtype="float32")(h)

    return keras.Model(inputs=inputs, outputs=logits)

    
if __name__ == "__main__":
    import config_vit_base_96 as conf
    
    classifier_model = vit_classifier(
        orig_image_shape=(32, 32, 3),
        image_shape=conf.IMAGE_SHAPE,
        patch_size=conf.PATCH_SIZE,
        num_layers=conf.NUM_LAYERS,
        num_heads=conf.NUM_HEADS,
        mlp_dim=conf.MLP_DIM,
        attention_dropout_rate=conf.ATTENTION_DROPOUT_RATE,
        dropout_rate=conf.DROPOUT_RATE,
        num_classes=conf.NUM_CLASSES
    )
    
    print(classifier_model.summary())

