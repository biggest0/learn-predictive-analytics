# =============================================================================
# SMOOTHING & HOLT-WINTERS CHEATSHEET
#   Simple Exponential Smoothing  → level only (no trend, no seasonality)
#   Holt's Linear               → level + trend
#   Holt-Winters                → level + trend + seasonality
# =============================================================================

import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
df = pd.read_csv(
    "https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv",
    header=0).squeeze()

# -----------------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------
TEST_STEPS = 10
y_train = df[0:len(df) - TEST_STEPS] # all rows - last n rows
y_test  = df[len(df) - TEST_STEPS:] # last n rows (kept as test data)

# -----------------------------------------------------------------------------
# 3. SIMPLE EXPONENTIAL SMOOTHING  (level only)
#    alpha : smoothing factor — high = more weight on recent values
# -----------------------------------------------------------------------------
holt_single = SimpleExpSmoothing(y_train).fit(smoothing_level=0.2, optimized=False, use_brute=True) # alpha could be 1/(2*m)
holt_single_forecast = holt_single.forecast(TEST_STEPS)

rmse = sqrt(mean_squared_error(y_test, holt_single_forecast))
print('Simple Exponential Smoothing RMSE: %.3f' % rmse)

# -----------------------------------------------------------------------------
# 4. HOLT'S LINEAR (double exponential smoothing — level + trend)
#    alpha : level smoothing  |  beta : trend smoothing
# -----------------------------------------------------------------------------
holt_double = ExponentialSmoothing(y_train, trend='add').fit()
holt_double_forecast = holt_double.forecast(TEST_STEPS)

rmse = sqrt(mean_squared_error(y_test, holt_double_forecast))
print("Holt's Linear RMSE: %.3f" % rmse)

# -----------------------------------------------------------------------------
# 5. HOLT-WINTERS (triple exponential smoothing — level + trend + seasonality)
#    trend       : 'add' or 'mul'
#    seasonal    : 'add' or 'mul'
#    seasonal_periods : length of one season cycle (e.g. 12=monthly, 7=weekly)
# -----------------------------------------------------------------------------
holt_triple = ExponentialSmoothing(
    y_train,
    trend='add',
    seasonal='add',
    seasonal_periods=12,    # ← change to match your data's seasonality
).fit()
holt_triple_forecast = holt_triple.forecast(TEST_STEPS)

rmse = sqrt(mean_squared_error(y_test, holt_triple_forecast))
print('Holt-Winters RMSE: %.3f' % rmse)

print(holt_triple.summary())

# -----------------------------------------------------------------------------
# 6. PLOT ALL FORECASTS vs ACTUAL
# -----------------------------------------------------------------------------
plt.plot(y_test.values,       label='Actual',      marker='o')
plt.plot(holt_single_forecast.values, label='SES', marker='o')
plt.plot(holt_double_forecast.values, label="Holt's", marker='o')
plt.plot(holt_triple_forecast.values, label='Holt-Winters', marker='o')
plt.legend()
plt.title("Smoothing Forecasts vs Actual")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# KEY CONCEPTS (exam reminders)
# -----------------------------------------------------------------------------
# SES           : one smoothing param (alpha) — no trend or seasonality
# Holt's        : adds beta for trend — good for trended, non-seasonal data
# Holt-Winters  : adds gamma for seasonality — best for seasonal data
# trend='add'   : additive trend (constant increase)
# trend='mul'   : multiplicative trend (growing increase)
# seasonal='add': additive seasonality (constant seasonal swings)
# seasonal='mul': multiplicative seasonality (growing seasonal swings)
# seasonal_periods: how many time steps in one full season cycle
# optimized=True: lets statsmodels find best alpha/beta/gamma automatically
