# Dav-assignment1
# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# ================================
# 2. LOAD DATA
# ================================
df = pd.read_csv("Quote-Equity-BHEL-EQ-18-03-2025-18-03-2026.csv")

print(df.head())  # check columns

# ================================
# 3. CLEAN COLUMN NAMES
# ================================
df.columns = df.columns.str.strip()

# Rename if needed (depends on NSE format)
if 'Close Price' in df.columns:
    df.rename(columns={'Close Price': 'Close'}, inplace=True)
elif 'CLOSE' in df.columns:
    df.rename(columns={'CLOSE': 'Close'}, inplace=True)

if 'Date' not in df.columns:
    df.rename(columns={'DATE': 'Date'}, inplace=True)

# ================================
# 4. DATA PREPROCESSING
# ================================

# Convert Date (NSE uses DD-MM-YYYY)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Sort + index
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# Handle missing values
df['Close'] = df['Close'].fillna(method='ffill')

# ================================
# 5. VISUALIZATION
# ================================
plt.figure(figsize=(10,5))
plt.plot(df['Close'])
plt.title("BHEL Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid()
plt.show()

# ================================
# 6. ADF TEST
# ================================
result = adfuller(df['Close'])

print("\nADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("Data is Stationary")
else:
    print("Data is NOT Stationary")

# ================================
# 7. DIFFERENCING
# ================================
df_diff = df['Close'].diff().dropna()

# ================================
# 8. ACF & PACF
# ================================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plot_acf(df_diff, lags=20, ax=plt.gca())

plt.subplot(1,2,2)
plot_pacf(df_diff, lags=20, ax=plt.gca())

plt.show()

# ================================
# 9. ARIMA MODEL
# ================================
model = ARIMA(df['Close'], order=(1,1,1))
model_fit = model.fit()

print(model_fit.summary())

# ================================
# 10. FORECAST 30 DAYS
# ================================
forecast = model_fit.forecast(steps=30)

# Create future dates
future_dates = pd.date_range(start=df.index[-1], periods=31, freq='D')[1:]

forecast_df = pd.DataFrame({'Forecast': forecast}, index=future_dates)

# ================================
# 11. PLOT FORECAST
# ================================
plt.figure(figsize=(10,5))

plt.plot(df['Close'], label='Historical')
plt.plot(forecast_df['Forecast'], label='Forecast', color='red')

plt.title("BHEL Price Forecast (30 Days)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()

plt.show()

# ================================
# 12. INTERPRETATION
# ================================
print("\nInterpretation:")
print("If forecast line trends upward → Uptrend")
print("If downward → Downtrend")
print("If flat → Stability")
