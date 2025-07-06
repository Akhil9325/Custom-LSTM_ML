import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Load the saved results
with open('image_shape_results.pkl', 'rb') as f:
    results = pickle.load(f)

rnn_results = results['rnn_results']
lstm_results = results['lstm_results']
image_shapes = results['image_shapes']
shape_labels = [f"{shape[0]}x{shape[1]}" for shape in image_shapes]

# Plot confusion matrices
def plot_confusion_matrix(true_labels, predictions, model_name):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower()}.png', dpi=300)
    plt.close()

# Generate confusion matrix plots
for i, shape in enumerate(image_shapes):
    shape_label = f"{shape[0]}x{shape[1]}"
    
    # RNN confusion matrix
    plot_confusion_matrix(
        results['rnn_true_labels'][i], 
        results['rnn_predictions'][i], 
        f'RNN-{shape_label}'
    )
    
    # LSTM confusion matrix
    plot_confusion_matrix(
        results['lstm_true_labels'][i], 
        results['lstm_predictions'][i], 
        f'LSTM-{shape_label}'
    )

# 1. Training Loss vs Epochs (RNN vs LSTM) - separate graph for each image shape
for i, shape in enumerate(image_shapes):
    shape_label = f"{shape[0]}x{shape[1]}"
    plt.figure(figsize=(10, 6))
    
    # Plot RNN and LSTM losses for this specific image shape
    plt.plot(rnn_results[shape_label]['train_losses'], 
             label=f'RNN', color='blue', linewidth=2)
    plt.plot(lstm_results[shape_label]['train_losses'], 
             label=f'LSTM', color='green', linewidth=2)
    
    plt.title(f'Training Loss vs Epochs (Image Shape: {shape_label})')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'training_loss_{shape_label}.png', dpi=300)
    plt.close()

# 2. Accuracy vs Image Shape (both models on same graph)
plt.figure(figsize=(12, 6))

# Extract final validation and training accuracies
rnn_val_accuracies = [rnn_results[f"{shape[0]}x{shape[1]}"]['final_val_accuracy'] for shape in image_shapes]
lstm_val_accuracies = [lstm_results[f"{shape[0]}x{shape[1]}"]['final_val_accuracy'] for shape in image_shapes]
rnn_train_accuracies = [rnn_results[f"{shape[0]}x{shape[1]}"]['final_train_accuracy'] for shape in image_shapes]
lstm_train_accuracies = [lstm_results[f"{shape[0]}x{shape[1]}"]['final_train_accuracy'] for shape in image_shapes]

x = np.arange(len(shape_labels))

# Plot accuracies vs image shape
plt.plot(x, rnn_val_accuracies, 'o-', label='RNN Validation', color='blue', linewidth=2)
plt.plot(x, lstm_val_accuracies, 's-', label='LSTM Validation', color='green', linewidth=2)
plt.plot(x, rnn_train_accuracies, 'o--', label='RNN Training', color='blue', alpha=0.6)
plt.plot(x, lstm_train_accuracies, 's--', label='LSTM Training', color='green', alpha=0.6)

plt.title('Accuracy vs Image Shape (RNN vs LSTM)')
plt.xlabel('Image Shape')
plt.ylabel('Accuracy (%)')
plt.xticks(x, shape_labels)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_vs_image_shape.png', dpi=300)
plt.close()

print("All plots have been generated and saved.")