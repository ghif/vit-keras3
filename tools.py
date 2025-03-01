import numpy as np
import matplotlib.pyplot as plt
from keras import ops

def display_patches(patches, patch_size, dimensions):
    """
    Display a set of image patches.
    Args:
        patches (Tensor): A tensor where the patches are stored. The shape of the tensor
            is `(num_patches, patch_size, patch_size, 3)`.
        patch_size (int): The size of the patches.
        dimensions (int): The number of rows and columns to use for the plot.
    """
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = ops.reshape(patch, (patch_size, patch_size, dimensions))
        plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"))
        plt.axis("off")

    plt.show()