import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Fetch historical data for TCS from NSE
symbol = 'TCS.NS'
data = yf.download(symbol, period='1d', interval='1m')

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Prepare training data
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 1  # Change time step to 1 minute
X, Y = create_dataset(scaled_data, time_step)
X_train = X.reshape(X.shape[0], X.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y, batch_size=1, epochs=50)

# Predicting the next 5 minutes
predictions = []
last_minute_data = scaled_data[-time_step:]
for _ in range(5):
    X_test = last_minute_data.reshape(1, -1, 1)
    predicted_stock_price = model.predict(X_test)
    predictions.append(predicted_stock_price[0][0])
    last_minute_data = np.append(last_minute_data[1:], predicted_stock_price)

# Inverse scaling
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Generate time index for the next 5 minutes
next_5_minutes_index = pd.date_range(data.index[-1], periods=6, freq='T')[1:]

# Plotting the results for the last hour only and the next 5 minutes
last_hour_data = data[-60:]  # Filter last hour of data
plt.figure(figsize=(14, 7))
plt.plot(last_hour_data.index, last_hour_data['Close'], label='Actual Close Price', color='blue')
plt.plot(next_5_minutes_index, predicted_prices, label='Predicted Price (next 5 minutes)', color='red', marker='o')
plt.title('TCS Stock Price Prediction (Next 5 Minutes)')
plt.xlabel('Datetime')
plt.ylabel('Price')
plt.legend()
plt.show()
