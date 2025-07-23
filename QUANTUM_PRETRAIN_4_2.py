import time
import os
import copy
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
learning_rate = 0.05 ###Learning rate for benchmarking NN
num_epochs = 6
pretrain_loop=6
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
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)




# Training loop
 # To store test losses
train_losses = []  # (Optional) To track training losses per epoch
stored_weights = []  # Array to store weights after the 3rd epoch

classical_train_accuracies = []
classical_test_accuracies = []

for epoch in range(num_epochs):
    model.train()  # Ensure model is in training mode
    correct, total = 0, 0
    epoch_train_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        epoch_train_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Store train accuracy
    train_acc = 100 * correct / total
    classical_train_accuracies.append(train_acc)

    # Store weights after 3rd epoch
    if epoch == pretrain_loop - 1:
        stored_weights = [param.detach().cpu().numpy() for param in model.parameters()]
        print(f"The pretraining for the QT will happen till epoch: {epoch}")

    # Testing loop
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device),labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate test accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and store test accuracy
    test_acc = 100 * correct / total
    classical_test_accuracies.append(test_acc)

    # Print train and test accuracy for each epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

numpy_weights = {}
nw_list = []
nw_list_normal = []
for name, param in model.state_dict().items():
    numpy_weights[name] = param.cpu().numpy()
for i in numpy_weights:
    nw_list.append(list(numpy_weights[i].flatten()))
for i in nw_list:
    for j in i:
        nw_list_normal.append(j)
print("# of NN parameters: ", len(nw_list_normal))
optima_weights=model.state_dict()
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
        self.MappingNetwork = self.MappingModel(n_qubit + 1, [4,4], 1).to(device)

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
        return x, prob_val_post_processed


# Instantiate the model, move it to GPU, and set up loss function and optimizer
hybrid_model = LewHybridNN().to(device)
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(hybrid_model.parameters(), lr=0.02)

# for param in model.MappingNetwork.parameters():
#     param.requires_grad=False

# exp_lr_scheduler = lr_scheduler.StepLR(
#     optimizer, step_size=10, gamma=gamma_lr_scheduler
# )


num_trainable_params_QNN = sum(p.numel() for p in LewHybridNN.MappingModel(n_qubit+1,  [4,2], 1).parameters() if p.requires_grad)

num_trainable_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
print("# of trainable parameter in Mapping model: ", num_trainable_params_QNN)
print("# of trainable parameter in QNN model: ", num_trainable_params - num_trainable_params_QNN)
print("# of trainable parameter in full model: ", num_trainable_params)

# Initialize a list to store test accuracies
test_accuracies = []
# Compare the weights and biases
# Training loop with test accuracy tracking
# Convert stored weights to a tensor
stored_weights = torch.tensor(np.concatenate([w.flatten() for w in stored_weights]), dtype=torch.float32).to(device)
print(f"stored_weights_shape: {stored_weights.shape}")

# Training loop with test accuracy tracking
for epoch in range(50):
    hybrid_model.train()  # Set model to training mode
    for i, (images, labels) in enumerate(train_loader):
        since_batch = time.time()

        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()

        # Forward pass
        outputs, QT_weights = hybrid_model(images)
        if epoch==2:
            print(f"Epoch {epoch + 1}, Batch {i + 1}:")
            print(f"Stored Weights Shape: {stored_weights.shape}")
            print(f"Generated QT Weights Shape: {QT_weights.squeeze().shape}")


        # Print shapes of stored weights and QT_weights


        # Compute loss
        loss = criterion(stored_weights, QT_weights.squeeze())

        # Print MSE loss
        print(f"MSE Loss: {loss.item()}")

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    # Evaluate on the test set after each epoch





criterion1=nn.CrossEntropyLoss()
optimizer = optim.Adam(hybrid_model.parameters(), lr=0.008)
print(f"stored_weight:{stored_weights}")
print(f"generated_weights:{QT_weights.squeeze()}")
train_accuracy_qt = []
test_accuracy_qt = []
hybrid_num_epoch=100
# for epoch in range(num_epochs - pretrain_loop):
for epoch in range(hybrid_num_epoch):
    hybrid_model.train()  # Set model to training mode
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs, QT_weights = hybrid_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Compute loss
        loss = criterion1(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Calculate train accuracy
    train_acc = 100 * correct / total
    train_accuracy_qt.append(train_acc)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Accuracy: {train_acc:.2f}%")

    # Evaluate on test set
    hybrid_model.eval()  # Set model to evaluation mode
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = hybrid_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and store test accuracy
    test_acc = 100 * correct / total
    test_accuracy_qt.append(test_acc)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Test Accuracy: {test_acc:.2f}%")

total_test_accuracy=classical_test_accuracies+test_accuracy_qt
print(f"total_test_10:{total_test_accuracy[0:10]}")
#print(f"length of test_accuracy_qt{len(test_accuracy_qt)},length of classical_test_accuracies{len(classical_test_accuracies)}")
new_hybrid_model=LewHybridNN().to(device)
criterion1=nn.CrossEntropyLoss()
optimizer = optim.Adam(new_hybrid_model.parameters(), lr=0.07)

train_accuracy_lew = []  # Renamed
test_accuracy_lew = []  # Renamed

for epoch in range(hybrid_num_epoch+num_epochs):
    new_hybrid_model.train()  # Set model to training mode
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs, QT_weights = new_hybrid_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Compute loss
        loss = criterion1(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Calculate train accuracy
    train_acc = 100 * correct / total
    train_accuracy_lew.append(train_acc)  # Updated variable
    print(f"Epoch [{epoch + 1}/{hybrid_num_epoch}] Train Accuracy: {train_acc:.2f}%")

    # Evaluate on test set
    new_hybrid_model.eval()  # Set model to evaluation mode
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = new_hybrid_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and store test accuracy
    test_acc = 100 * correct / total
    test_accuracy_lew.append(test_acc)  # Updated variable
    print(f"Epoch [{epoch + 1}/{hybrid_num_epoch}] Test Accuracy: {test_acc:.2f}%")


data1 = {
    "Epoch": list(range(len(total_test_accuracy))),
    "Pretrained_Accuracy": total_test_accuracy,
    "Quantum_Train_Accuracy": test_accuracy_lew,
}

df1 = pd.DataFrame(data1)
df1.to_csv("QT_Pretrain_beam.csv", index=False)
plt.figure(figsize=(8,6))
plt.plot(total_test_accuracy, marker='o', linestyle='-', color='black', label="Pretrained Method")
plt.plot(test_accuracy_lew, marker='s', linestyle='--', color='black', label="Quantum Train")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("Pretrain_test_accuracy_plot.png", dpi=300)
plt.show()
plt.figure(figsize=(8, 6))

