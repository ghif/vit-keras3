import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
import keras_hub

import dataset

# Constants
BASE_MODEL = "vit_base_patch16_224_imagenet"
MODEL_PATH = "models/vit_base_224_finetuned_cifar100.weights.h5"
BATCH_SIZE = 128
IMAGE_SHAPE = (224, 224, 3)
TOP_K = 5

# Use mixed precision
keras.mixed_precision.set_global_policy("mixed_float16")

# Load CIFAR100 dataset
train_dataset, test_dataset, dataset_info = dataset.prepare_cifar100(BATCH_SIZE, IMAGE_SHAPE, st_type=-1, augment=False)
num_classes = dataset_info.features["label"].num_classes
class_names = dataset_info.features["label"].names
print(f"Class names: {class_names}")

# Load trained Keras model
preprocessor = keras_hub.models.ViTImageClassifierPreprocessor.from_preset(
    BASE_MODEL
)
backbone = keras_hub.models.Backbone.from_preset(BASE_MODEL)
image_classifier = keras_hub.models.ViTImageClassifier(
    backbone=backbone,
    num_classes=num_classes,
    preprocessor=preprocessor,
)

# Set DType Policy float32 for last layer
last_layer = image_classifier.layers[-1]
last_layer.dtype_policy = keras.mixed_precision.Policy("float32")
# 

print(image_classifier.summary(expand_nested=True))

# Check layer dtype policies
for i, layer in enumerate(image_classifier.layers): 
    print(f"[{i}] {layer.name} - {layer.dtype_policy}")

image_classifier.load_weights(MODEL_PATH)


def model_predict(processed_image):
    logits = image_classifier.predict(processed_image)
    probs = tf.nn.softmax(logits, axis=-1) # Convert to probabilities
    return probs
    

def predict_image(input_image: Image.Image):
    """
    Takes a PIL Image, preprocesses it, predicts the class, and returns the label.
    """
    print(f"Received image of type: {type(input_image)}")
    if input_image is None:
        return "Please upload an image."

    # Preprocess the image
    img_array = np.array(input_image)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_resized = tf.image.resize(img_tensor, size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]), method='lanczos5')

    # Add batch dimension (model expects batch_size, height, width, channels)
    img_batch = tf.expand_dims(img_resized, axis=0)

    # Get predictions
    predictions = model_predict(img_batch) 

    # Get indices of top 5 probability
    top_indices = tf.argsort(predictions, axis=-1, direction='DESCENDING')[0][:TOP_K].numpy()
    top_indices = top_indices.tolist()

    # Construct a dictionary of class names and their corresponding probabilities
    output_dict = {}
    for i in range(len(top_indices)):
        class_name = class_names[top_indices[i]]
        class_prob = predictions[0][top_indices[i]].numpy()
        output_dict[class_name] = class_prob
    
    return output_dict

# --- Create Gradio Interface ---
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Image"), # Input is a PIL Image
    outputs=gr.Label(num_top_classes=TOP_K, label="Prediction"), # Output is a Label component
    title="ViT Image Classification Demo",
    description="Upload an image and the model will predict its class.",
    examples=[
        # Add paths to example images if you have any
        # ["path/to/example1.jpg"],
        # ["path/to/example2.png"],
    ],
    allow_flagging="never" # Optional: Disables flagging
)

# --- Launch the UI ---
if __name__ == "__main__":
    print("Launching Gradio interface...")
    # share=True creates a public link (use with caution)
    demo.launch(share=False)