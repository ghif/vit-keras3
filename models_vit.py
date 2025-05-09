import keras
from keras import layers
from keras import ops
    
def extract_patches(images, patch_size):
    """
    Extract patches from a batch of images.

    Args:
        images (Tensor): A batch of images with shape (batch_size, height, width, channels).
        patch_size (int): The size of the patches to be extracted.
    Returns:
        Tensor: The extracted patches.
    """
    (_, height, width, channels) = ops.shape(images)
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    patches = ops.image.extract_patches(
        images,
        size=patch_size
    )
    patches = keras.layers.Reshape((num_patches_h * num_patches_w, patch_size * patch_size * channels))(patches)
    return patches

def encode_patches(patch, num_patches, projection_dim):
    """
    Encode a single image patch.

    Args:
        patch (Tensor): A single image patch with shape (patch_dim).
        num_patches (int): Number of patches the image is divided into.
        projection_dim (int): Dimension of the projection space.
    Returns:
        Tensor: The encoded patch.
    """
    positions = ops.expand_dims(
        ops.arange(start=0, stop=num_patches, step=1), axis=0
    )
    projection = layers.Dense(units=projection_dim)(patch)
    position_embedding = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)
    return projection + position_embedding

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

# def augment_and_resize(images, image_size):
#     """
#     Apply data augmentation and resize the input images.

#     Args:
#         images (Tensor): A batch of images with shape (batch_size, height, width, channels).
#         image_size (int): The size of the output images.
#     Returns:
#         Tensor: The augmented and resized batch of images.
#     """
#     z = layers.Normalization()(images)
#     z = layers.Resizing(image_size, image_size)(z)
#     z = layers.RandomFlip("horizontal")(z)
#     z = layers.RandomRotation(factor=0.02)(z)
#     z = layers.RandomZoom(height_factor=0.2, width_factor=0.2)(z)
#     return z

# def normalize_and_resize(images, image_size):
#     """
#     Normalize and resize the input images.

#     Args:
#         images (Tensor): A batch of images with shape (batch_size, height, width, channels).
#         image_size (int): The size of the output images.
#     Returns:
#         Tensor: The normalized and resized batch of images.
#     """
#     z = layers.Normalization()(images)
#     z = layers.Resizing(image_size, image_size)(z)
#     return z

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
    x = layers.LayerNormalization(epsilon=1e-3)(inputs)

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
    x = x + inputs

    # MLP block
    y = layers.LayerNormalization(epsilon=1e-3)(x)
    y = mlp_block(y, mlp_dim, dropout_rate)
    
    # Skip connection 2
    y = y + x
    
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

def vit_backbone(image_shape, patch_size, num_layers, num_heads, mlp_dim, attention_dropout_rate, dropout_rate):
    """
    Create a Vision Transformer backbone.

    Args:
        image_shape (tuple): Shape of the images after resizing and augmentation.
        patch_size (int): Size of the patches.
        num_layers (int): Number of encoder layers.
        num_heads (int): Number of attention heads.
        mlp_dim (int): Hidden dimension of the MLP block.
        attention_dropout_rate (float): Dropout rate for the attention layer.
        dropout_rate (float): Dropout rate for the block.
    Returns:
        Vision Transformer backbone (keras.Model).
    """
    num_patches = (image_shape[0] // patch_size) * (image_shape[1] // patch_size)
    hidden_dim = patch_size * patch_size * 3

    inputs = keras.Input(shape=image_shape)
    # augmented = augment_and_resize(inputs, image_shape[0])
    # augmented = normalize_and_resize(inputs, image_shape[0])
    # preprocessed_inputs = preprocess(inputs)
    patches = extract_patches(inputs, patch_size)
    encoded_patches = encode_patches(patches, num_patches, hidden_dim)

    y = vit_encoder(
        encoded_patches, num_layers, num_heads, hidden_dim, mlp_dim, attention_dropout_rate, dropout_rate
    )
    return keras.Model(inputs=inputs, outputs=y)

def vit_classifier(image_shape, patch_size, num_layers, num_heads, mlp_dim, attention_dropout_rate, dropout_rate, num_classes):
    """
    Create a Vision Transformer classifier using ViT Backbone.

    Args:
        image_shape (tuple): Shape of the images after resizing and augmentation.
        patch_size (int): Size of the patches.
        num_layers (int): Number of encoder layers.
        num_heads (int): Number of attention heads.
        mlp_dim (int): Hidden dimension of the MLP block.
        attention_dropout_rate (float): Dropout rate for the attention layer.
        dropout_rate (float): Dropout rate for the block.
        num_classes (int): Number of output classes.
    Returns:
        Vision Transformer classifier (keras.Model).
    """
    backbone = vit_backbone(
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
    import json
    import numpy as np

    config_path = "config_vit_base_96_train.json"

    # Load configuration from JSON file
    with open(config_path) as f:
        conf = json.load(f)
    
    IMAGE_SHAPE = tuple(conf["image_shape"])
    PATCH_SIZE = conf["patch_size"]
    NUM_LAYERS = conf["num_layers"]
    NUM_HEADS = conf["num_heads"]
    MLP_DIM = conf["mlp_dim"]
    ATTENTION_DROPOUT_RATE = conf["attention_dropout_rate"]
    DROPOUT_RATE = conf["dropout_rate"]
    NUM_CLASSES = conf["num_classes"]

    LEARNING_RATE = conf["learning_rate"]
    WEIGHT_DECAY = conf["weight_decay"]
    GLOBAL_CLIPNORM = conf["global_clipnorm"]
    BATCH_SIZE = conf["batch_size"]
    
    classifier_model = vit_classifier(
        image_shape=IMAGE_SHAPE,
        patch_size=PATCH_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        attention_dropout_rate=ATTENTION_DROPOUT_RATE,
        dropout_rate=DROPOUT_RATE,
        num_classes=NUM_CLASSES
    )
    
    print(classifier_model.summary())
    keras.utils.plot_model(classifier_model, to_file="vit_base_96_cifar100.png", expand_nested=True, show_shapes=True)

    optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    global_clipnorm=GLOBAL_CLIPNORM,
)

    classifier_model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    for i, layer in enumerate(classifier_model.layers):
        print(f"[{i}] {layer.name} - {layer.dtype_policy}")

