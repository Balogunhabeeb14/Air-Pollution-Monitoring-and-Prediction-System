import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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

# Define the hyperparameters to tune
param_grid = {
    'batch_size': [16, 32],
    'epochs': [50, 100],
    'units': [64, 128],
    'dropout_rate': [0.2, 0.3]
}

best_rmse = float('inf')
best_params = {}

# Loop through the parameter combinations
for batch_size in param_grid['batch_size']:
    for epochs in param_grid['epochs']:
        for units in param_grid['units']:
            for dropout_rate in param_grid['dropout_rate']:
                # Create and compile the model
                model = Sequential()
                model.add(LSTM(units=units, return_sequences=True, input_shape=(look_back, X_train.shape[2])))
                model.add(Dropout(dropout_rate))
                model.add(LSTM(units=units, return_sequences=True))
                model.add(Dropout(dropout_rate))
                model.add(LSTM(units=units))
                model.add(Dense(units=int(units/2), activation='relu'))
                model.add(Dense(units=1))  # Output layer
                model.compile(optimizer='adam', loss='mean_squared_error')
                
                # Train the model
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                
                # Predict on test data
                y_pred = model.predict(X_test)
                combined_data = np.concatenate((X_test[:, -1, :], y_pred), axis=1)
                inverse_transformed_data = scaler.inverse_transform(combined_data)
                inverse_transformed_no2 = inverse_transformed_data[:, -1]
                
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(data['NO2'].values[train_size + look_back:], inverse_transformed_no2))
                
                # Check if this model's RMSE is better than the best so far
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'units': units,
                        'dropout_rate': dropout_rate
                    }

# Print the best hyperparameters found
print("Best RMSE:", best_rmse)
print("Best Hyperparameters:", best_params)
