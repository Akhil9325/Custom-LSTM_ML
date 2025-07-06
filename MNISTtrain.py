import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
import os

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# Custom LSTM Implementation
# --------------------------------------------------
class VerboseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Input gate parameters
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        # Forget gate parameters
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        # Cell gate parameters
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))
        # Output gate parameters
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, state):
        h_prev, c_prev = state
        i_t = torch.matmul(x, self.W_ii.t()) + torch.matmul(h_prev, self.W_hi.t()) + self.b_i
        i_t = torch.sigmoid(i_t)
        f_t = torch.matmul(x, self.W_if.t()) + torch.matmul(h_prev, self.W_hf.t()) + self.b_f
        f_t = torch.sigmoid(f_t)
        g_t = torch.matmul(x, self.W_ig.t()) + torch.matmul(h_prev, self.W_hg.t()) + self.b_g
        g_t = torch.tanh(g_t)
        o_t = torch.matmul(x, self.W_io.t()) + torch.matmul(h_prev, self.W_ho.t()) + self.b_o
        o_t = torch.sigmoid(o_t)
        c_next = f_t * c_prev + i_t * g_t
        h_next = o_t * torch.tanh(c_next)
        return h_next, (h_next, c_next)

# Load MNIST dataset
def load_mnist_data(batch_size=512):  # Increased batch size for speed
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

class VanillaRNN(nn.Module):
    def __init__(self, seq_length, feature_size, hidden_size, num_classes=10):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.rnn = nn.RNN(feature_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        # Reshape based on sequence length and feature size
        x = x.view(batch_size, self.seq_length, self.feature_size)
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])
        return output

class CustomLSTMModel(nn.Module):
    def __init__(self, seq_length, feature_size, hidden_size, num_classes=10):
        super(CustomLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.lstm = VerboseLSTM(feature_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        # Reshape based on sequence length and feature size
        x = x.view(batch_size, self.seq_length, self.feature_size)
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)
        for t in range(self.seq_length):
            x_t = x[:, t, :]
            h, (h, c) = self.lstm(x_t, (h, c))
        output = self.fc(h)
        return output

def train_model(model_type, seq_length, feature_size, hidden_size=32, epochs=5, batch_size=512):
    train_loader, test_loader = load_mnist_data(batch_size)
    if model_type == 'rnn':
        model = VanillaRNN(seq_length=seq_length, feature_size=feature_size, hidden_size=hidden_size).to(device)
    elif model_type == 'lstm':
        model = CustomLSTMModel(seq_length=seq_length, feature_size=feature_size, hidden_size=hidden_size).to(device)
    else:
        raise ValueError("model_type must be 'rnn' or 'lstm'")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
        epoch_train_loss /= len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()
        epoch_val_loss /= len(test_loader.dataset)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_accuracy)

        print(f" Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    training_time = time.time() - start_time
    final_train_loss = train_losses[-1]
    final_train_accuracy = train_accuracies[-1]
    final_val_loss = val_losses[-1]
    final_val_accuracy = val_accuracies[-1]
    print(f" Final training loss: {final_train_loss:.4f}, accuracy: {final_train_accuracy:.2f}%")
    print(f" Final validation loss: {final_val_loss:.4f}, accuracy: {final_val_accuracy:.2f}%")
    print(f" Training time: {training_time:.2f} seconds")

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'final_train_loss': final_train_loss,
        'final_train_accuracy': final_train_accuracy,
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_accuracy,
        'training_time': training_time
    }

# Function to collect predictions for confusion matrices
def collect_predictions(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.numpy())
    return all_preds, all_labels

# Different image shapes to test (seq_length, feature_size)
image_shapes = [
    (28, 28),   # Original 28x28 images
    (56, 14),   # Reshape to 56x14
    (112, 7)    # Reshape to 112x7
]

# Keep hidden size constant
hidden_size = 32
epochs = 10  # Number of epochs

# Store results
rnn_results = {}
lstm_results = {}

# Load test loader once
_, test_loader = load_mnist_data(batch_size=512)

# Create dictionary to store all results
results = {
    'rnn_results': {},
    'lstm_results': {},
    'image_shapes': image_shapes,
    'rnn_predictions': [],
    'rnn_true_labels': [],
    'lstm_predictions': [],
    'lstm_true_labels': []
}

for shape in image_shapes:
    seq_length, feature_size = shape
    print(f"\nTraining with image shape {seq_length}x{feature_size}...")

    print(f" Training RNN...")
    rnn_model = VanillaRNN(seq_length=seq_length, feature_size=feature_size, hidden_size=hidden_size).to(device)
    rnn_results[shape] = train_model('rnn', seq_length=seq_length, feature_size=feature_size, hidden_size=hidden_size, epochs=epochs, batch_size=512)
    results['rnn_results'][f"{seq_length}x{feature_size}"] = rnn_results[shape]
    
    # Collect RNN predictions for confusion matrix
    rnn_preds, rnn_labels = collect_predictions(rnn_model, test_loader)
    results['rnn_predictions'].append(rnn_preds)
    results['rnn_true_labels'].append(rnn_labels)

    print(f" Training custom LSTM...")
    lstm_model = CustomLSTMModel(seq_length=seq_length, feature_size=feature_size, hidden_size=hidden_size).to(device)
    lstm_results[shape] = train_model('lstm', seq_length=seq_length, feature_size=feature_size, hidden_size=hidden_size, epochs=epochs, batch_size=512)
    results['lstm_results'][f"{seq_length}x{feature_size}"] = lstm_results[shape]
    
    # Collect LSTM predictions for confusion matrix
    lstm_preds, lstm_labels = collect_predictions(lstm_model, test_loader)
    results['lstm_predictions'].append(lstm_preds)
    results['lstm_true_labels'].append(lstm_labels)

# Save results to file
with open('image_shape_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n===== Performance Summary =====")
print(f"{'Image Shape':<15} {'RNN Train Acc':<15} {'RNN Val Acc':<15} {'Custom LSTM Train Acc':<20} {'Custom LSTM Val Acc':<20}")
print("-" * 90)
for shape in image_shapes:
    seq_length, feature_size = shape
    shape_key = f"{seq_length}x{feature_size}"
    rnn_train_acc = rnn_results[shape]['final_train_accuracy']
    rnn_val_acc = rnn_results[shape]['final_val_accuracy']
    lstm_train_acc = lstm_results[shape]['final_train_accuracy']
    lstm_val_acc = lstm_results[shape]['final_val_accuracy']
    print(f"{shape_key:<15} {rnn_train_acc:<15.2f}% {rnn_val_acc:<15.2f}% {lstm_train_acc:<20.2f}% {lstm_val_acc:<20.2f}%")

print(f"\nTraining complete. Results saved to 'image_shape_results.pkl'")
print("You can modify plot_results.py to visualize these results.")