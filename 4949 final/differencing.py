# =============================================================================
# STATIONARITY & DIFFERENCING CHEATSHEET
# =============================================================================

import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
df = pd.read_csv(
    "https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv",
    names=['value'], header=0)
df = df['value']

df.plot()
plt.title("Original Series")
plt.show()

# -----------------------------------------------------------------------------
# 2. TEST FOR STATIONARITY — ADF Test
#    H0 : non-stationary  |  reject H0 if p-value < 0.05
# -----------------------------------------------------------------------------
result = adfuller(df.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
# p > 0.05 → non-stationary → apply differencing

# -----------------------------------------------------------------------------
# 3. FIRST-ORDER DIFFERENCING  (removes linear trend)
# -----------------------------------------------------------------------------
diff1 = df.diff(1) # 1 means lag1, 2 would mean subtract 2 steps back

result_diff1 = adfuller(diff1.dropna()) # remove NaN, or adfuller will break
print('\nAfter 1st-Order Differencing:')
print('ADF Statistic: %f' % result_diff1[0]) # ADF statistics, more negative means stationary -4 and smaller
print('p-value: %f' % result_diff1[1]) # p-value
print('Critical Values:', result_diff1[4])

diff1.plot()
plt.title("1st-Order Differenced")
plt.show()

# -----------------------------------------------------------------------------
# 4. SECOND-ORDER DIFFERENCING  (if diff1 still non-stationary)
# -----------------------------------------------------------------------------
diff2 = df.diff(1).diff(1) # do 2 rounds of differencing

result_diff2 = adfuller(diff2.dropna())
print('\nAfter 2nd-Order Differencing:')
print('ADF Statistic: %f' % result_diff2[0])
print('p-value: %f' % result_diff2[1])
print('Critical Values:', result_diff2[4])

# -----------------------------------------------------------------------------
# 5. ACF / PACF PLOTS  (help identify AR and MA terms for ARIMA)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2)
plot_acf(diff1.dropna(), ax=axes[0])
plot_pacf(diff1.dropna(), ax=axes[1])
plt.show()

# -----------------------------------------------------------------------------
# KEY CONCEPTS (exam reminders)
# -----------------------------------------------------------------------------
# ADF H0         : non-stationary — reject (stationary) if p < 0.05
# diff(1)        : y_t - y_{t-1}  removes linear trend
# diff(2)        : apply diff(1) twice — removes quadratic trend
# ACF plot       : identifies MA(q) order — cuts off after lag q
# PACF plot      : identifies AR(p) order — cuts off after lag p
# d in ARIMA(p,d,q) : number of differences needed for stationarity


# Add this to Cheatsheet
df = pd.read_csv(
    "https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv", header=0)
df = df['x']

df.plot()
plt.title("Original Series")
plt.show()

result = adfuller(df.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

diff1 = df.diff(1) # 1 means lag1, 2 would mean subtract 2 steps back
# diff2 = df.diff(1).diff(1) # do 2 rounds of differencing

result_diff1 = adfuller(diff1.dropna()) # remove NaN, or adfuller will break
print('\nAfter 1st-Order Differencing:')
print('ADF Statistic: %f' % result_diff1[0]) # ADF statistics, more negative means stationary -4 and smaller
print('p-value: %f' % result_diff1[1]) # p-value
print('Critical Values:', result_diff1[4]) # compare with 5% for 95 CI, if smaller means stationary else diff

diff1.plot()
plt.title("1st-Order Differenced")
plt.show()