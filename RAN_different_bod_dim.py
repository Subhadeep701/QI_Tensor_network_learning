# Define the MPSWithDense model with bond_dim 3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorkrowch as tk
import matplotlib.pyplot as plt
import pandas as pd
# Select device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load Diabetes dataset
# data = load_diabetes()
# X = data.data  # Features
# y = data.target.reshape(-1, 1)
# minmax_scaler = MinMaxScaler()
# y = minmax_scaler.fit_transform(y)  # Scale y to range [0, 1]
# input_size = X.shape[1]  # Number of features
# output_dim = 1  # Regression output
#
# # Normalize the features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
#
# # Convert data to PyTorch tensors
# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.float32)
#
# # Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
#
# # Create TensorDatasets
# train_dataset = TensorDataset(X_train, y_train)
# test_dataset = TensorDataset(X_test, y_test)
#
# # DataLoader Parameters
# batch_size = 50
#
# # Create DataLoaders
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
combined_df = pd.read_csv("C:/Users/emnusob/OneDrive - Ericsson/Desktop/Research/Saravanan sir/Patent RAN/CODES/pythonProject4/combined_data.csv")

# Select features (X) and target (y) from the specified range
X = combined_df.drop(columns=['dl_brate']).iloc[1000:3000]
y = combined_df['dl_brate'].iloc[1000:3000]

# Initialize scalers for X and y
scaler_X = MinMaxScaler()  # Standard scaling for features
scaler_y = MinMaxScaler()    # MinMax scaling for target (range [0, 1])

# Scale X and y
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1)  # Ensure y is shaped as (N, 1)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# DataLoader Parameters
batch_size = 500
input_size = X.shape[1]  # Number of features
output_dim = 1  # Regression output
# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Output the shapes for verification
print(f"Train X shape: {X_train.shape}, Train y shape: {y_train.shape}")
print(f"Test X shape: {X_test.shape}, Test y shape: {y_test.shape}")

# Model hyperparameters
embedding_dim = 2
bond_dim = 2
init_method = 'randn_eye'
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 60

# Define the model with MPS and Dense Layer
class MPSWithDense(nn.Module):
    def __init__(self, input_size, embedding_dim, bond_dim, output_dim):
        super(MPSWithDense, self).__init__()
        self.mps_layer = tk.models.MPSLayer(
            n_features=input_size + 1,
            in_dim=embedding_dim,
            out_dim=8,
            bond_dim=bond_dim,
            boundary='obc',
            init_method=init_method,
            std=1e-9,
            device=device
        )
        self.dense = nn.Sequential(
            nn.Linear(8, 24),
            nn.Linear(24, output_dim)
        )

    def forward(self, x):
        x = tk.embeddings.poly(x, degree=embedding_dim - 1)
        x = self.mps_layer(x, inline_input=False, inline_mats=False, renormalize=False)
        x = self.dense(x)
        return x

# Initialize the MPS model
model = MPSWithDense(input_size, embedding_dim, bond_dim, output_dim).to(device)
model.mps_layer.trace(
    torch.zeros(1, input_size, embedding_dim, device=device),
    inline_input=False,
    inline_mats=False,
    renormalize=False
)

# Define the Feed-Forward Neural Network Model
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_dim):
        super(FeedForwardNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Initialize the Feed-Forward model
ffn_model = FeedForwardNN(input_size, hidden_size1=32, hidden_size2=24, output_dim=output_dim).to(device)

# Function to calculate the number of trainable parameters

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
ffn_optimizer = optim.Adam(ffn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Lists to store test losses for plotting
test_loss_mps = []
test_loss_ffn = []

# Train and evaluate the MPSWithDense model
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(data)
        loss = criterion(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_loss = sum(criterion(model(data.to(device)), targets.to(device)).item() for data, targets in test_loader) / len(test_loader)
    test_loss_mps.append(test_loss)

# Train and evaluate the FeedForwardNN model
for epoch in range(num_epochs):
    ffn_model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = ffn_model(data)
        loss = criterion(predictions, targets)

        # Backward pass
        ffn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ffn_model.parameters(), max_norm=1.0)
        ffn_optimizer.step()

    # Evaluate on test set
    ffn_model.eval()
    with torch.no_grad():
        test_loss = sum(criterion(ffn_model(data.to(device)), targets.to(device)).item() for data, targets in test_loader) / len(test_loader)
    test_loss_ffn.append(test_loss)


class MPSWithDenseBond3(nn.Module):
    def __init__(self, input_size, embedding_dim, bond_dim, output_dim):
        super(MPSWithDenseBond3, self).__init__()
        self.mps_layer = tk.models.MPSLayer(
            n_features=input_size + 1,
            in_dim=embedding_dim,
            out_dim=8,
            bond_dim=4,
            boundary='obc',
            init_method=init_method,
            std=1e-9,
            device=device
        )
        self.dense = nn.Sequential(
            nn.Linear(8, 24),
            nn.Linear(24, output_dim)
        )

    def forward(self, x):
        x = tk.embeddings.poly(x, degree=embedding_dim - 1)
        x = self.mps_layer(x, inline_input=False, inline_mats=False, renormalize=False)
        x = self.dense(x)
        return x

# Initialize the MPS model with bond_dim 3
model_bond3 = MPSWithDenseBond3(input_size, embedding_dim, bond_dim=3, output_dim=output_dim).to(device)
model_bond3.mps_layer.trace(
    torch.zeros(1, input_size, embedding_dim, device=device),
    inline_input=False,
    inline_mats=False,
    renormalize=False
)

# Define the optimizer for the new model
optimizer_bond3 = optim.Adam(model_bond3.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Lists to store test losses for plotting
test_loss_mps_bond3 = []

# Train and evaluate the MPSWithDense model with bond_dim 3
for epoch in range(num_epochs):
    model_bond3.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model_bond3(data)
        loss = criterion(predictions, targets)

        # Backward pass
        optimizer_bond3.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_bond3.parameters(), max_norm=1.0)
        optimizer_bond3.step()

    # Evaluate on test set
    model_bond3.eval()
    with torch.no_grad():
        test_loss = sum(criterion(model_bond3(data.to(device)), targets.to(device)).item() for data, targets in test_loader) / len(test_loader)
    test_loss_mps_bond3.append(test_loss)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print parameter complexity for both models
mps_params = count_parameters(model)
mps_params_bond_up = count_parameters(model_bond3)
ffn_params = count_parameters(ffn_model)

print(f"Number of trainable parameters in MPSWithDense: {mps_params}")
print(f"Number of trainable parameters in FeedForwardNN: {ffn_params}")
print(f"Number of trainable parameters in MPSWithDense: {mps_params_bond_up}")

# Plotting the test losses for both models
# Plotting the test losses for both models with an interval of 4 epochs
plt.figure(figsize=(10, 6))
epoch_diff=5
start=15
plt.plot(range(start, num_epochs + 1, epoch_diff), test_loss_mps[start-1:num_epochs:epoch_diff], label="Hybrid MPS Model (bond dimension 2) Test Loss", color='black', marker='o', linestyle='--')
plt.plot(range(start, num_epochs + 1, epoch_diff), test_loss_ffn[start-1:num_epochs:epoch_diff], label="Neural Network Model Test Loss", color='black', marker='s', linestyle='-')
plt.plot(range(start, num_epochs + 1, epoch_diff), test_loss_mps_bond3[start-1:num_epochs:epoch_diff], label="Hybrid MPS Model (bond dimension 4) Test Loss", color='black', marker='x', linestyle='-.')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Test Loss vs. Epoch for Different Models ")
plt.legend()
plt.grid(True)
plt.show()
