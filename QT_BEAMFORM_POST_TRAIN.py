import time
import os
import copy
import random
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
# Pennylane
import pennylane as qml
from pennylane import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Plotting
import matplotlib.pyplot as plt


import itertools

# OpenMP: number of parallel threads.
# os.environ["OMP_NUM_THREADS"] = "1"

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Data loading and preprocessing for CIFAR-10
batch_size = 1500
learning_rate = 0.05  ###Learning rate for benchmarking NN
num_epochs = 150
q_depth = 1# Depth of the quantum circuit
q_delta = 0.4  # Initial spread of random quantum weights
step = 0.085        # Learning rate for the tunable VQC
gamma_lr_scheduler = 0.1    # Learning rate reduction applied every 10 epochs.

# Load IRIS dataset

# Load and preprocess the dataset
train = pd.read_excel("C:/Users/emnusob/OneDrive - Ericsson/Desktop/Research/Beam_forming/best_parameters_datahand.xlsx")

# Extract labels and features
labels = train.iloc[:, 5].values - 1
#features = train.iloc[:, [9, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28, 31, 32]].values
#features = train.iloc[:, [9, 10, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28, 31, 32]].values
features = train.iloc[:, [9, 10, 11, 15, 19, 23, 27, 31]].values

# Normalize features
scaler = MinMaxScaler()
#scaler=StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training (90%) and testing (10%) sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader objects  # Define the batch size
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 8)
        self.fc2 = nn.Linear(8, len(torch.unique(y_train)))  # Input features: 10, Output features: 3 (3 classes in IRIS)

    def forward(self, x):
        x = self.fc1(x)
        x=torch.relu(x)
        x = self.fc2(x)
        return x



# Instantiate the model, move it to GPU, and set up loss function and optimizer
model1 = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=learning_rate)




# Training loop
classical_test_accuracies = []  # To store test losses
train_losses = []  # (Optional) To track training losses per epoch

for epoch in range(num_epochs):
    model1.train()  # Ensure model is in training mode
    epoch_train_loss = 0  # Accumulate training loss for the epoch

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()

        # Forward pass
        outputs = model1(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        epoch_train_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Average training loss for the epoch (optional)
    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)  # Track training loss (if needed)

    # Testing loop at the end of the epoch
    model1.eval()  # Ensure model is in evaluation mode
    test_loss = 0  # To accumulate test loss
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU

            # Forward pass
            outputs = model1(images)

            # Compute test loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Average test loss for the epoch
      # Store test loss

    # Calculate accuracy
    accuracy = 100 * correct / total
    classical_test_accuracies.append(accuracy)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

print(f"Accuracy on the test set: {(100 * correct / total):.2f}%")
numpy_weights = {}
nw_list = []
nw_list_normal = []
for name, param in model1.state_dict().items():
    numpy_weights[name] = param.cpu().numpy()
for i in numpy_weights:
    nw_list.append(list(numpy_weights[i].flatten()))
for i in nw_list:
    for j in i:
        nw_list_normal.append(j)
print("# of NN parameters: ", len(nw_list_normal))
optima_weights=model1.state_dict()
#print(f"optimal weights:{optima_weights}")
n_qubits = int(np.ceil(np.log2(len(nw_list_normal))))
print("Required qubit number: ", n_qubits)

dev = qml.device("default.qubit", wires=n_qubits)
#dev = qml.device("lightning.gpu", wires=n_qubits, batch_obs=True)


n_qubit = n_qubits


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def RZ_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RZ(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


def probs_to_weights(probs_):
    new_state_dict = {}
    data_iterator = probs_.view(-1)

    for name, param in SimpleCNN().state_dict().items():
        shape = param.shape
        num_elements = param.numel()
        chunk = data_iterator[:num_elements].reshape(shape)
        new_state_dict[name] = chunk
        data_iterator = data_iterator[num_elements:]

    return new_state_dict


def generate_qubit_states_torch(n_qubit):
    # Create a tensor of shape (2**n_qubit, n_qubit) with all possible combinations of 0 and 1
    all_states = torch.cartesian_prod(*[torch.tensor([-1, 1]) for _ in range(n_qubit)])
    return all_states


####################



@qml.qnode(dev, diff_method="spsa")
# @qml.qnode(dev, diff_method="parameter-shift")

def quantum_net(q_weights_flat):
    """
    The variational quantum circuit.
    """
    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)
    H_layer(n_qubits)
    # Repeated layer
    for i in range(q_depth):
        # Parameterised layer
        if i % 2 == 0:
            for y in range(n_qubits):
                qml.RY(q_weights[i][y], wires=y)
        else:
            for z in range(n_qubits):
                qml.RZ(q_weights[i][z], wires=z)
        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y + 1])

    # state_mag = qml.probs(wires=list(range(n_qubits)))

    return qml.probs(wires=list(range(n_qubits)))  # x_


class LewHybridNN(nn.Module):
    class MappingModel(nn.Module):
        def __init__(self, input_size, hidden_sizes, output_size):
            super().__init__()
            self.input_layer = nn.Linear(input_size, hidden_sizes[0])
            self.hidden_layers = nn.ModuleList([
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)
            ])
            self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        def forward(self, X):
            X = X.type_as(self.input_layer.weight)
            X = self.input_layer(X)
            X=torch.relu(X)
            for hidden in self.hidden_layers:
                X = hidden(X)
                #X= torch.relu(X)
            return self.output_layer(X)

    def __init__(self):
        super().__init__()
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
        self.MappingNetwork = self.MappingModel(n_qubit + 1, [2], 1).to(device)

    def forward(self, x):
        device = x.device
        self.q_params.requires_grad = True
        easy_scale_coeff = 2 ** (n_qubit - 1)
        gamma, beta, alpha = 0.1, 0.8, 0.3

        probs_ = quantum_net(self.q_params)
        x_ = (beta * torch.tanh(gamma * easy_scale_coeff * torch.abs(probs_) ** 2)) ** alpha
        x_ = x_ - torch.mean(x_)
        x_ = x_.to(device)

        qubit_states_torch = generate_qubit_states_torch(n_qubit).to(device)
        combined_data_torch = torch.cat((qubit_states_torch, x_.unsqueeze(1)), dim=1)
        prob_val_post_processed = self.MappingNetwork(combined_data_torch)[:len(nw_list_normal)]
        state_dict = probs_to_weights(prob_val_post_processed)

        dtype = torch.float32
        x = F.linear(x, state_dict['fc1.weight'].to(device).type(dtype), state_dict['fc1.bias'].to(device).type(dtype))
        x=torch.relu(x)
        x = F.linear(x, state_dict['fc2.weight'].to(device).type(dtype), state_dict['fc2.bias'].to(device).type(dtype))
        return x, state_dict


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
model = LewHybridNN().to(device)  # Hybrid Model
simple_model = SimpleCNN().to(device)  # Simple CNN for switched training
num_trainable_params_QNN = sum(p.numel() for p in LewHybridNN.MappingModel(n_qubit+1,  [2], 1).parameters() if p.requires_grad)

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("# of NN parameters: ", len(nw_list_normal))
print("# of trainable parameter in Mapping model: ", num_trainable_params_QNN)
print("# of trainable parameter in QNN model: ", num_trainable_params - num_trainable_params_QNN)
print("# of trainable parameter in full model: ", num_trainable_params)
criterion = nn.CrossEntropyLoss()
optimizer_hybrid = optim.Adam(model.parameters(), lr=step)
optimizer_simple = optim.Adam(simple_model.parameters(), lr=0.085)

# Initialize losses
train_losses_hybrid, test_losses_hybrid = [], []
train_losses_switched, test_losses_switched = [], []

training_switch = False  # Keep OFF initially
switch_epoch = None  # To track the switching epoch
QT_weights = None  # Placeholder for saved quantum weights

print("\n===== Training Both Methods (Hybrid & Switched) =====")
for epoch in range(num_epochs):
    model.train()
    running_loss_hybrid = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_hybrid.zero_grad()
        outputs, QT_weights = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_hybrid.step()
        running_loss_hybrid += loss.item()

    train_loss_hybrid = running_loss_hybrid / len(train_loader)
    train_losses_hybrid.append(train_loss_hybrid)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss (Hybrid): {train_loss_hybrid:.4f}")

    # Evaluation for hybrid model
    model.eval()
    correct_hybrid, total_hybrid = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_hybrid += targets.size(0)
            correct_hybrid += (predicted == targets).sum().item()

    accuracy_hybrid = 100 * correct_hybrid / total_hybrid
    test_losses_hybrid.append(accuracy_hybrid)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy (Hybrid): {accuracy_hybrid:.2f}%")

    # Check for switching condition
    if accuracy_hybrid >= 80 and not training_switch and QT_weights is not None:
        print("Accuracy reached 60%! Switching to SimpleNN.")
        training_switch = True
        switch_epoch = epoch+1
        simple_model.load_state_dict(QT_weights, strict=True)
    training_switch = switch_epoch is not None and epoch >= switch_epoch
    if not training_switch:
        # Store same values in switched losses since no switching yet
        train_losses_switched.append(train_loss_hybrid)
        test_losses_switched.append(accuracy_hybrid)
    else:
        # Training SimpleCNN after switching epoch
        simple_model.train()
        running_loss_simple = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_simple.zero_grad()
            outputs = simple_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_simple.step()
            running_loss_simple += loss.item()

        train_loss_simple = running_loss_simple / len(train_loader)
        train_losses_switched.append(train_loss_simple)
        print(f"Switched method Epoch [{epoch + 1}/{num_epochs}], Train Loss (SimpleNN): {train_loss_simple:.4f}")

        # Evaluation for SimpleCNN
        simple_model.eval()
        correct_simple, total_simple = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = simple_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_simple += targets.size(0)
                correct_simple += (predicted == targets).sum().item()

        accuracy_simple = 100 * correct_simple / total_simple
        test_losses_switched.append(accuracy_simple)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy (SimpleNN): {accuracy_simple:.2f}%")

# Save results for comparison
saved_test_losses_hybrid = test_losses_hybrid[:]
saved_test_losses_switched = test_losses_switched[:]

# Plot the losses after training


# Create a DataFrame
data = {
    "Epochs": num_epochs+1,
    "Classical_NN_Accuracy": classical_test_accuracies,
    "Quantum_Train_Accuracy": saved_test_losses_hybrid,
    "Mixed_Training_Accuracy": saved_test_losses_switched
}
df = pd.DataFrame(data)

# Save to an Excel file
output_path = "Post_trainingn_QT_beam.xlsx"
df.to_excel(output_path, index=False)

print(f"Data saved to {output_path}")



# Define the x values with a step of 3
epochs = list(range(1, num_epochs + 1, 3))

plt.figure(figsize=(10, 6))

# Classical Neural Network
plt.plot(epochs, classical_test_accuracies[0::3],
         label='Classical Neural Network', marker='d', linestyle='--',
         c='black', markersize=4)

# Quantum Update
plt.plot(epochs, saved_test_losses_hybrid[0::3],
         label='Quantum Train', marker='s', linestyle='-',
         c='black', markersize=4)

# Mixed Training
plt.plot(epochs, saved_test_losses_switched[0::3],
         label='Mixed Training', marker='o', linestyle='-.',
         c='black', markersize=4)

# Labels and legend
plt.xlabel("Epochs")
plt.ylabel("Accuracy/Loss")
plt.title("Comparison of Training Methods")
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)


plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig("Mixed_Train_bemclass_less_feature.png", dpi=300)
plt.show()

