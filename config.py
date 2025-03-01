# Configuration dictionary for Vision Transformer
vit_config = {
    # Training parameters
    "training": {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 256,
        "num_epochs": 10,  # For real training, use num_epochs=100. 10 is a test value
    },
    
    # Image parameters
    "image": {
        "image_size": 72,  # We'll resize input images to this size
        "patch_size": 6,  # Size of the patches to be extract from the input images
        "num_patches": (72 // 6) ** 2,  # Calculated from image_size and patch_size
    },
    
    # Model architecture parameters
    "model": {
        # Transformer parameters
        "projection_dim": 64,
        "num_heads": 4,
        "transformer_units": [128, 64],  # Size of the transformer layers
        "transformer_layers": 8,
        
        # MLP head parameters
        "mlp_head_units": [2048, 1024],  # Size of the dense layers of the final classifier
    }
}