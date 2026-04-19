"""fits the model once on all training data, then forecasts n steps into the future (beyond the known data). Those predictions have no ground truth to compare against yet."""
# =============================================================================
# ARIMA CHEATSHEET  —  ARIMA(p, d, q)
#   p = AR order  (lags of y, from PACF)
#   d = differencing order (from ADF test)
#   q = MA order  (lags of error, from ACF)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
df = pd.read_csv(
    "https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv",
    header=0).squeeze()

df.plot()
plt.title("Original Series")
plt.show()

# -----------------------------------------------------------------------------
# 2. CHECK STATIONARITY  → determines d
# -----------------------------------------------------------------------------
result = adfuller(df.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:', result[4])
# p > 0.05 → non-stationary → set d=1

# -----------------------------------------------------------------------------
# 3. ACF / PACF ON DIFFERENCED SERIES  → determines p and q
#    PACF cuts off after lag p  → AR(p)
#    ACF  cuts off after lag q  → MA(q)
# -----------------------------------------------------------------------------
diff1 = df.diff(1)
fig, axes = plt.subplots(1, 2)
plot_acf(diff1.dropna(), ax=axes[0])
plot_pacf(diff1.dropna(), ax=axes[1])
plt.show()

# -----------------------------------------------------------------------------
# 4. BUILD ARIMA MODEL  ← swap (p, d, q) based on ACF/PACF above
# -----------------------------------------------------------------------------
model = ARIMA(df, order=(1, 1, 1))   # (p=1, d=1, q=1)
model_fit = model.fit()

print(model_fit.summary())

# -----------------------------------------------------------------------------
# 5. RESIDUAL DIAGNOSTICS  (residuals should look like white noise)
# -----------------------------------------------------------------------------
residuals = model_fit.resid

fig, axes = plt.subplots(1, 2)
residuals.plot(ax=axes[0], title="Residuals")
residuals.plot(ax=axes[1], kind='kde', title="Residual Density")
plt.show()

fig, axes = plt.subplots(1, 2)
plot_acf(residuals, ax=axes[0], title="ACF of Residuals")
plot_pacf(residuals, ax=axes[1], title="PACF of Residuals")
plt.show()

# -----------------------------------------------------------------------------
# 6. FORECAST
# -----------------------------------------------------------------------------
n_steps = 10
forecast = model_fit.forecast(steps=n_steps)
print('\nForecast (next %d steps):' % n_steps)
print(forecast)

# Plot forecast vs actuals
plt.plot(df.values, label='Actual')
plt.plot(range(len(df), len(df) + n_steps), forecast, label='Forecast', color='red')
plt.title("ARIMA Forecast")
plt.legend()
plt.show()

# -----------------------------------------------------------------------------
# KEY CONCEPTS (exam reminders)
# -----------------------------------------------------------------------------
# p (AR)   : autoregressive order — read from PACF, lag where it cuts off
# d        : differencing order — number of diffs to make series stationary
# q (MA)   : moving average order — read from ACF, lag where it cuts off
# AIC/BIC  : lower is better — use summary() to compare models
# Residuals: should be white noise (no pattern in ACF, ~normal distribution)
# forecast : predicts n_steps ahead; uncertainty grows with horizon
