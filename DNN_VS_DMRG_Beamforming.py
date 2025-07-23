import time
import torch
import numpy as np
import pandas as pd
from torchmps import MPS
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import InputLayer
import matplotlib.pyplot as plt

# Load data
train = pd.read_excel("C:/Users/emnusob/OneDrive - Ericsson/Desktop/Research/Beam_forming/best_parameters_datahand.xlsx")

# Extract labels and features
labels = train.iloc[:, 5].values - 1
train_x = train.iloc[:, [9, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28, 31, 32]].values

feature_matrix = preprocessing.normalize(train_x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.1, random_state=42)

# Convert labels to categorical (one-hot encoding)
num_classes = np.unique(y_train).shape[0]
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Define the Keras model with multiple hidden layers
dnn_model = Sequential()
dnn_model.add(InputLayer(shape=(x_train.shape[1],)))
dnn_model.add(Dense(24, activation='relu'))
dnn_model.add(Dense(12, activation='relu'))
dnn_model.add(Dense(num_classes, activation='softmax'))


dnn_model.compile(optimizer=Adam(learning_rate=0.02), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = dnn_model.fit(x_train, y_train, epochs=100, batch_size=500, validation_data=(x_test, y_test))

# Evaluate the model
dnn_test_accuracies = history.history['val_accuracy']
dnn_final_loss, dnn_final_accuracy = dnn_model.evaluate(x_test, y_test)
print(f"DNN Test loss: {dnn_final_loss:.4f}")
print(f"DNN Test accuracy: {dnn_final_accuracy:.4f}")


single_hidden_layer_model = Sequential()
single_hidden_layer_model.add(InputLayer(shape=(x_train.shape[1],)))
single_hidden_layer_model.add(Dense(24, activation='relu'))
single_hidden_layer_model.add(Dense(num_classes, activation='softmax'))

# Compile the model
single_hidden_layer_model.compile(optimizer=Adam(learning_rate=0.02), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_single_hidden_layer = single_hidden_layer_model.fit(x_train, y_train, epochs=100, batch_size=500, validation_data=(x_test, y_test))

# Evaluate the model
single_hidden_layer_test_accuracies = history_single_hidden_layer.history['val_accuracy']
single_hidden_layer_final_loss, single_hidden_layer_final_accuracy = single_hidden_layer_model.evaluate(x_test, y_test)
print(f"Single Hidden Layer DNN Test loss: {single_hidden_layer_final_loss:.4f}")
print(f"Single Hidden Layer DNN Test accuracy: {single_hidden_layer_final_accuracy:.4f}")

# MPS Model
features_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
labels_tensor = torch.tensor(labels.squeeze(), dtype=torch.long)

unique_labels = torch.unique(labels_tensor)
output_dim = unique_labels.size(0)

bond_dim = 3
adaptive_mode = False
periodic_bc = False

num_train = len(train)
num_test = int(0.2 * num_train)
batch_size = 500
num_epochs = 100
learn_rate = 1e-2
l2_reg = 0.0

input_dim = features_tensor.shape[1]
mps = MPS(
    input_dim=input_dim,
    output_dim=output_dim,
    bond_dim=bond_dim,
    adaptive_mode=adaptive_mode,
    periodic_bc=periodic_bc,
)

loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mps.parameters(), lr=learn_rate, weight_decay=l2_reg)

dataset = TensorDataset(features_tensor, labels_tensor)

train_size = num_train - num_test
train_dataset, test_dataset = random_split(dataset, [train_size, num_test])

loaders = {
    "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
    "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
}
num_batches = {
    "train": len(loaders["train"]),
    "test": len(loaders["test"])
}

print(
    f"Training on {train_size} custom dataset samples \n"
    f"(testing on {num_test}) for {num_epochs} epochs"
)
print(f"Maximum MPS bond dimension = {bond_dim}")
print(f" * {'Adaptive' if adaptive_mode else 'Fixed'} bond dimensions")
print(f" * {'Periodic' if periodic_bc else 'Open'} boundary conditions")
print(f"Using Adam w/ learning rate = {learn_rate:.1e}")
if l2_reg > 0:
    print(f" * L2 regularization = {l2_reg:.2e}")
print()

train_losses = []
train_accuracies = []
mps_test_accuracies = []

for epoch_num in range(1, num_epochs + 1):
    running_loss = 0.0
    running_acc = 0.0
    start_time = time.time()
    for inputs, labels in loaders["train"]:
        inputs, labels = inputs.view([batch_size, input_dim]), labels.data

        scores = mps(inputs)

        _, preds = torch.max(scores, 1)

        loss = loss_fun(scores, labels)
        with torch.no_grad():
            accuracy = torch.sum(preds == labels).item() / batch_size
            running_loss += loss
            running_acc += accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = running_loss / num_batches['train']
    avg_train_acc = running_acc / num_batches['train']
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_acc)

    with torch.no_grad():
        running_acc = 0.0

        for inputs, labels in loaders["test"]:
            inputs, labels = inputs.view([batch_size, input_dim]), labels.data

            scores = mps(inputs)
            _, preds = torch.max(scores, 1)
            running_acc += torch.sum(preds == labels).item() / batch_size

    avg_test_acc = running_acc / num_batches['test']
    mps_test_accuracies.append(avg_test_acc)

    print(f"### Epoch {epoch_num} ###")
    print(f"Average loss:           {avg_train_loss:.4f}")
    print(f"Average train accuracy: {avg_train_acc:.4f}")
    print(f"Test accuracy:          {avg_test_acc:.4f}")
    print(f"Runtime:                {int(time.time()-start_time)} sec\n")

data = {
    "Epoch": list(range(1, num_epochs + 1)),
    "2-hidden layer NN": dnn_test_accuracies,
    "Single Hidden Layer NN": single_hidden_layer_test_accuracies,
    "Quantum Inspired Method": mps_test_accuracies,
}
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
output_file = "beamform_dnn_vs_dmrg.xlsx"
df.to_excel(output_file, index=False)

print(f"Data successfully saved to {output_file}")
# Plotting test accuracy for all three models
plt.figure(figsize=(10, 5))

plt.plot(range(1, num_epochs + 1), dnn_test_accuracies, label='2-hidden layer NN',marker='*', linestyle='-', color='cyan')
plt.plot(range(1, num_epochs + 1), single_hidden_layer_test_accuracies, label='Single Hidden Layer DNN Test Accuracy', marker='^', linestyle='--', color='magenta')
plt.plot(range(1, num_epochs + 1), mps_test_accuracies, label='Quantum Inspired Method Test Accuracy',marker='o', linestyle=':', color='green')

plt.xlabel('Epoch',fontsize=12)
plt.ylabel('Test Accuracy',fontsize=12)
#plt.title('Test Accuracy over Epochs for 2-hidden layer NN, Single Hidden Layer NN, and Quantum Inspired')
#plt.title('Test Accuracy over Epochs for 2-hidden layer NN, and Quantum Inspired')
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig('DMRG_VS_NN.pdf', format='pdf', dpi=300)
plt.tight_layout()
plt.show()
