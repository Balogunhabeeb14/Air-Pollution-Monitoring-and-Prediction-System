import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load your CSV file
data = pd.read_csv('/Users/habeeb/Downloads/AQI/NO2.csv')
data = data[['Timestamp', 'Humidity', 'Temp', 'speed', 'green_area', 'road_area', 'buildings', 'NO2']]
print(data.isna().sum())

# Before splitting the data
data['NO2'].fillna(data['NO2'].mean(), inplace=True)

# Convert date column to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Humidity', 'Temp', 'speed', 'green_area', 'road_area', 'buildings', 'NO2']])

# Create lag features
look_back = 5  # Number of previous time steps to use as features
X, y = [], []
for i in range(len(data_scaled) - look_back):
    X.append(data_scaled[i:i + look_back, :-1])
    y.append(data_scaled[i + look_back, -1])  # Target is No2

X, y = np.array(X), np.array(y)

# Train-test split
train_size = int(0.8 * len(data_scaled))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define TCN model
class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        )
        
    def forward(self, x):
        return self.tcn(x)

input_size = X_train.shape[2]
num_channels = 64
kernel_size = 3
dropout = 0.2
model = TCN(input_size, num_channels, kernel_size, dropout)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5
batch_size = 16
# Inside the training loop
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i + batch_size]
        targets = y_train[i:i + batch_size]
        
        optimizer.zero_grad()
        outputs = model(inputs.transpose(1, 2))
        
        # Expand the dimensions of targets to match the output dimensions
        targets_expanded = targets.unsqueeze(1).expand(-1, look_back)
        
        loss = criterion(outputs, targets_expanded)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test.transpose(1, 2))
    test_outputs_last = test_outputs[:, -1]  # Get the last predicted value of each sequence

# Convert test_outputs_last to a NumPy array
test_outputs_last_np = test_outputs_last.numpy()

# Flatten the test_outputs_last_np to match the shape of y_test
test_outputs_flat = test_outputs_last_np.flatten()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(data['NO2'].values[train_size + look_back: train_size + look_back + len(test_outputs_flat)], test_outputs_flat))
print(f'RMSE: {rmse}')


# Plot the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(data['Timestamp'].values[train_size + look_back:], data['NO2'].values[train_size + look_back:], label='Actual')
plt.plot(data['Timestamp'].values[train_size + look_back:], scaler.inverse_transform(test_outputs_flat.reshape(-1, 1)), label='Predicted', linestyle='dashed')
plt.xlabel('Timestamp')
plt.ylabel('NO2')
plt.title('Actual vs. Predicted NO2 Levels-TCN')
plt.legend()
plt.tight_layout()
plt.show()

# Create specificity and sensitivity chart
plt.figure(figsize=(6, 6))
plt.scatter(1 - specificity, sensitivity, color='blue')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve')
plt.show()
