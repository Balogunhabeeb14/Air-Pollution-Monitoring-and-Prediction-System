import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt

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
data_scaled = scaler.fit_transform(data[['Humidity',  'Temp', 'speed', 'green_area', 'road_area', 'buildings', 'NO2']])

# Check for NaN or Infinite values
nan_indices = np.where(np.isnan(data_scaled))
inf_indices = np.where(np.isinf(data_scaled))

if len(nan_indices[0]) > 0 or len(inf_indices[0]) > 0:
    print("Warning: NaN or Infinite values detected in the normalized data.")
    # Handle the issue or investigate the cause
if len(nan_indices[0]) > 0:
    print("NaN values detected in the following indices:")
    print(nan_indices)

if len(inf_indices[0]) > 0:
    print("Infinite values detected in the following indices:")
    print(inf_indices)   

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

# Build CNN-LSTM model with dropout layers
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))  # Add dropout layer here
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dropout(0.2))  # Add dropout layer here
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
plot_model(model, to_file='cnn_lstm_model.png', show_shapes=True)
# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test), verbose=2)

# Predict on test data
y_pred = model.predict(X_test)
# Concatenate X_test and y_pred along the last axis
combined_data = np.concatenate((X_test[:, -1, :], y_pred.reshape(-1, 1)), axis=1)

# Inverse transform the combined data using the scaler
inverse_transformed_data = scaler.inverse_transform(combined_data)

# Extract the last column (NO2 values)
inverse_transformed_no2 = inverse_transformed_data[:, -1]

#y_pred = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_pred.reshape(-1, 1)), axis=1))[:, -1]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(data['NO2'][train_size + look_back:], y_pred))
print(f'RMSE: {rmse}')

# Plot the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(data['Timestamp'].values[train_size + look_back:], data['NO2'].values[train_size + look_back:], label='Actual')
plt.plot(data['Timestamp'].values[train_size + look_back:], inverse_transformed_no2, label='Predicted', linestyle='dashed')
plt.xlabel('Timestamp')
plt.ylabel('NO2')
plt.title('Actual vs. Predicted NO2 Levels - CNN-LSTM')
plt.legend()
plt.tight_layout()
plt.show()
