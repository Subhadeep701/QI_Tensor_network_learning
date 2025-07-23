# Updated Code

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorkrowch as tk
import matplotlib.pyplot as plt
import pandas as pd

# Select device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load dataset
combined_df = pd.read_csv("C:/Users/emnusob/OneDrive - Ericsson/Desktop/Research/Saravanan sir/Patent RAN/CODES/pythonProject4/combined_data.csv")
X = combined_df.drop(columns=['dl_brate']).iloc[100:300]
y = combined_df['dl_brate'].iloc[100:300]

# Scale data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

# Define MPS Model
class MPSModel(nn.Module):
    def __init__(self, input_size, embedding_dim, bond_dim, output_dim):
        super(MPSModel, self).__init__()
        self.mps_layer = tk.models.MPSLayer(
            n_features=input_size + 1,
            in_dim=embedding_dim,
            out_dim=8,
            bond_dim=bond_dim,
            boundary='obc',
            init_method='randn_eye',
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

# Initialize Model
input_size = X.shape[1]
embedding_dim = 2
bond_dim = 2
output_dim = 1
model = MPSModel(input_size, embedding_dim, bond_dim, output_dim).to(device)
model.mps_layer.trace(torch.zeros(1, input_size, embedding_dim, device=device))

# Loss Functions
huber_criterion = nn.HuberLoss()
mse_criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Training with Huber Loss and MSE Loss
huber_losses = []
mse_losses = []

for epoch in range(40):
    model.train()
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)

        predictions = model(data)
        huber_loss = huber_criterion(predictions, targets)
        mse_loss = mse_criterion(predictions, targets)

        optimizer.zero_grad()
        huber_loss.backward()
        optimizer.step()

    # Evaluate losses on test set
    model.eval()
    with torch.no_grad():
        huber_test_loss = sum(huber_criterion(model(data.to(device)), targets.to(device)).item() for data, targets in test_loader) / len(test_loader)
        mse_test_loss = sum(mse_criterion(model(data.to(device)), targets.to(device)).item() for data, targets in test_loader) / len(test_loader)

    huber_losses.append(huber_test_loss)
    mse_losses.append(mse_test_loss)

# Plot Losses
plt.figure(figsize=(10, 6))
plt.plot(range(10, 41), huber_losses[9:], label="Huber Loss", color='black', marker='o', linestyle='--')
plt.plot(range(10, 41), mse_losses[9:], label="MSE Loss", color='black', marker='s', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Comparison of Huber Loss and MSE Loss (Epoch 10 onwards)")
plt.legend()
plt.grid(True, linestyle='--', color='gray', alpha=0.7)
plt.show()
