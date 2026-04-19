# =============================================================================
# REGRESSION WITH BACK-SHIFTED (LAG) FEATURES CHEATSHEET
# Use past values of a time series as features to predict the next value
# =============================================================================

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
# 2. CREATE BACK-SHIFTED (LAG) FEATURES
#    lag_1 = value from 1 step ago  (t-1)
#    lag_2 = value from 2 steps ago (t-2)
#    lag_3 = value from 3 steps ago (t-3)
#    shift(n) moves values DOWN by n rows — creates past observations as columns
# -----------------------------------------------------------------------------
data = pd.DataFrame()
data['y']     = df                  # target: value at time t
data['lag_1'] = df.shift(1)         # feature: value at t-1
data['lag_2'] = df.shift(2)         # feature: value at t-2
data['lag_3'] = df.shift(3)         # feature: value at t-3

data = data.dropna()                # drop rows with NaN from shifting
print(data.head(10))

# -----------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------
TEST_STEPS = 10

X = data[['lag_1', 'lag_2', 'lag_3']]
y = data['y']

X_train = X[0:len(X) - TEST_STEPS]
X_test  = X[len(X) - TEST_STEPS:]
y_train = y[0:len(y) - TEST_STEPS]
y_test  = y[len(y) - TEST_STEPS:]

# -----------------------------------------------------------------------------
# 4. LINEAR REGRESSION
# -----------------------------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, lr_preds))
print('Linear Regression RMSE: %.3f' % rmse)
print('Coefficients (lag_1, lag_2, lag_3):', lr.coef_)

# -----------------------------------------------------------------------------
# 5. PREDICT NEXT DAY (beyond the end of actual data)
#    Use the last 3 known values as lag features for the next unseen step
# -----------------------------------------------------------------------------
last_lag_1 = df.iloc[-1]    # most recent value
last_lag_2 = df.iloc[-2]    # one before that
last_lag_3 = df.iloc[-3]    # two before that

next_day_features = pd.DataFrame({'lag_1': [last_lag_1],
                                   'lag_2': [last_lag_2],
                                   'lag_3': [last_lag_3]})

next_day_pred = lr.predict(next_day_features)
print('Predicted next day value: %.3f' % next_day_pred[0])

# -----------------------------------------------------------------------------
# 6. PLOT ACTUAL vs PREDICTIONS
# -----------------------------------------------------------------------------
plt.plot(y_test.values,  label='Actual',             marker='o')
plt.plot(lr_preds,       label='Linear Regression',  marker='o')
plt.legend()
plt.title("Regression with Lag Features")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# KEY CONCEPTS (exam reminders)
# -----------------------------------------------------------------------------
# Lag feature    : past value used as a predictor — shift(n) creates lag of n steps
# shift(1)       : moves column down 1 row → t-1 value aligns with t target
# dropna()       : required after shifting — first n rows will have NaN
# Back-shift     : converting time series into supervised learning problem
# n lags         : how many past steps to include — tune based on ACF/PACF
# Any regressor  : lag features work with LinearRegression, RF, XGBoost, etc.
