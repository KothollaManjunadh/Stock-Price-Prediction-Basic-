# Stock Price Prediction using Linear Regression

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 1. Download Stock Data
# -------------------------------
stock_symbol = "AAPL"  # Apple Stock

df = yf.download(stock_symbol, start="2020-01-01", end="2025-01-01")

print(df.head())

# -------------------------------
# 2. Visualize Closing Prices
# -------------------------------
plt.figure(figsize=(12,5))
plt.plot(df['Close'])
plt.title(f"{stock_symbol} Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# -------------------------------
# 3. Create Target Column
# Predict Next Day Closing Price
# -------------------------------
df['Prediction'] = df['Close'].shift(-1)

# Remove last row (NaN target)
df = df[:-1]

# Features and Target
X = np.array(df[['Close']])
y = np.array(df['Prediction'])

# -------------------------------
# 4. Chronological Train-Test Split
# -------------------------------
split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# -------------------------------
# 5. Train Linear Regression Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 6. Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 7. Evaluation
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# -------------------------------
# 8. Plot Actual vs Predicted
# -------------------------------
plt.figure(figsize=(12,5))
plt.plot(y_test, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')
plt.title("Actual vs Predicted Stock Price")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 9. Predict Next Day Price
# -------------------------------
last_close = df[['Close']].tail(1)

next_day_price = model.predict(last_close)

print("\nLast Closing Price:", float(last_close.iloc[0]))
print("Predicted Next Day Price:", float(next_day_price[0]))
