import json
import matplotlib.pyplot as plt

# Open json history files
with open("models/vit_tiny_v2_cifar100_history.json", "r") as f:
    history_dict1 = json.load(f)

with open("models/vit_tiny_v2_cont_cifar100_history.json", "r") as f:
    history_dict2 = json.load(f)

# Combine the two history dictionaries
history_dict = {}
for key in history_dict1.keys():
    if key in history_dict2:
        # Concatenate the lists from both dictionaries
        history_dict[key] = history_dict1[key] + history_dict2[key]
    else:
        # If key only exists in history_dict1
        history_dict[key] = history_dict1[key]

# Add any keys that are only in history_dict2
for key in history_dict2.keys():
    if key not in history_dict1:
        history_dict[key] = history_dict2[key]

# Plot the training history
plt.figure(figsize=(12, 6))
# Plot training & validation accuracy
# plt.subplot(1, 2, 1)
plt.plot(history_dict['accuracy'], label='Training Accuracy')
plt.plot(history_dict['top-5-accuracy'], label='Training Top-5 Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
plt.plot(history_dict['val_top-5-accuracy'], label='Validation Top-5 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# # Plot training & validation loss
# plt.subplot(1, 2, 2)
# plt.plot(history_dict['loss'], label='Training Loss')
# plt.plot(history_dict['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig("img/vit_tiny_v2_cifar100_plot.png")

