import json
import matplotlib.pyplot as plt

# Open json history files
with open("models/vit_base_96_cifar100_history.json", "r") as f:
    history_dict = json.load(f)

# Plot the training history
plt.figure(figsize=(12, 6))
# Plot training & validation accuracy
# plt.subplot(1, 2, 1)
plt.plot(history_dict['accuracy'], label='Training Top-1 Accuracy')
plt.plot(history_dict['top-5-accuracy'], label='Training Top-5 Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Top-1 Accuracy')
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
plt.savefig("img/vit_base_96_cifar100_plot.png")

