# Configuration dictionary for Vision Transformer
vit_config = {
    # Training parameters
    "training": {
        "learning_rate": 0.0001,
        "weight_decay": 0.03,
        "batch_size": 128,
        "num_epochs": 100,  # For real training, use num_epochs=100. 10 is a test value
    },
    
    # Image parameters
    "image": {
        "image_size": 224,  # We'll resize input images to this size
        "patch_size": 16,  # Size of the patches to be extract from the input images
        "num_patches": (224 // 16) ** 2,  # Calculated from image_size and patch_size
    },
    
    # Model architecture parameters
    "model": {
        # Transformer parameters
        "projection_dim": 768,
        "transformer_layers": 2,
        "num_heads": 12,
        "mlp_size": 3072,  # Size of the transformer layers
        
        
        # # MLP head parameters
        # "mlp_head_units": [3072],  # Size of the dense layers of the final classifier
    }
}