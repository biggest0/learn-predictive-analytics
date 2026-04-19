# =============================================================================
# REGRESSION WITH BACK-SHIFTED (LAG) FEATURES CHEATSHEET
# Use past values of a time series as features to predict the next value
# =============================================================================

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
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
# 7. MULTIVARIATE LAG FEATURES (multiple input columns)
#    Same idea — shift each column separately, rename so you know which lag
#    e.g. temperature prediction with rain, humidity, wind as extra features
# -----------------------------------------------------------------------------
# Simulate a multivariate dataset (swap with your real CSV)
np.random.seed(42)
multi = pd.DataFrame({
    'temperature': df.values,
    'humidity':    np.random.uniform(40, 90, len(df)),
    'wind':        np.random.uniform(0, 30, len(df)),
    'rain':        np.random.uniform(0, 10, len(df)),
})

multi_data = pd.DataFrame()
multi_data['y'] = multi['temperature']   # target

# Create lag features for every column
for column in multi.columns: # all the original columns, times shifted 1,2,3 days
    multi_data[column + '_lag_1'] = multi[column].shift(1)
    multi_data[column + '_lag_2'] = multi[column].shift(2)
    multi_data[column + '_lag_3'] = multi[column].shift(3)

multi_data = multi_data.dropna() # drop first 3 rows that has NaN due to shifting
print("\nMultivariate lag feature columns:\n", multi_data.columns.tolist())

X_multi = multi_data.drop(columns=['y'])
y_multi = multi_data['y']

# create test and train split
X_multi_train = X_multi[0:len(X_multi) - TEST_STEPS]
X_multi_test  = X_multi[len(X_multi) - TEST_STEPS:]
y_multi_train = y_multi[0:len(y_multi) - TEST_STEPS]
y_multi_test  = y_multi[len(y_multi) - TEST_STEPS:]

lr_multi = LinearRegression()
lr_multi.fit(X_multi_train, y_multi_train)
lr_multi_preds = lr_multi.predict(X_multi_test)

rmse = sqrt(mean_squared_error(y_multi_test, lr_multi_preds))
print('Multivariate Linear Regression RMSE: %.3f' % rmse)

# Predict next day — use last known row of each column as lag_1, etc.
next_row = {}
for column in multi.columns:
    next_row[column + '_lag_1'] = [multi[column].iloc[-1]] # past 1 day, kept in list since its df standard
    next_row[column + '_lag_2'] = [multi[column].iloc[-2]] # past 2 day
    next_row[column + '_lag_3'] = [multi[column].iloc[-3]] # past 3 day

next_day_multi = pd.DataFrame(next_row) # makes 1 row of data to use as input for prediction
print('Multivariate predicted next day: %.3f' % lr_multi.predict(next_day_multi)[0]) # predict first row

# -----------------------------------------------------------------------------
# 8. LOGISTIC REGRESSION WITH LAG FEATURES
#    Same structure — only difference is target is binary (0 or 1)
#    and metrics are accuracy/classification_report instead of RMSE
# -----------------------------------------------------------------------------
# Create binary target: 1 if value went UP vs previous day, 0 if down
log_data = pd.DataFrame()
log_data['y']     = (df.diff(1) > 0).astype(int)   # 1 = up, 0 = down
log_data['lag_1'] = df.shift(1)
log_data['lag_2'] = df.shift(2)
log_data['lag_3'] = df.shift(3)

log_data = log_data.dropna()

X_log = log_data[['lag_1', 'lag_2', 'lag_3']]
y_log = log_data['y']

X_log_train = X_log[0:len(X_log) - TEST_STEPS]
X_log_test  = X_log[len(X_log) - TEST_STEPS:]
y_log_train = y_log[0:len(y_log) - TEST_STEPS]
y_log_test  = y_log[len(y_log) - TEST_STEPS:]

log_model = LogisticRegression()
log_model.fit(X_log_train, y_log_train)
log_preds = log_model.predict(X_log_test)

print('\nLogistic Regression Accuracy: %.3f' % accuracy_score(y_log_test, log_preds))
print(classification_report(y_log_test, log_preds, target_names=['Down', 'Up']))

# Predict next day direction
next_day_log = pd.DataFrame({'lag_1': [df.iloc[-1]],
                              'lag_2': [df.iloc[-2]],
                              'lag_3': [df.iloc[-3]]})
next_direction = log_model.predict(next_day_log)
print('Predicted next day direction: %s' % ('Up' if next_direction[0] == 1 else 'Down'))

# -----------------------------------------------------------------------------
# KEY CONCEPTS (exam reminders)
# -----------------------------------------------------------------------------
# Lag feature    : past value used as a predictor — shift(n) creates lag of n steps
# shift(1)       : moves column down 1 row → t-1 value aligns with t target
# dropna()       : required after shifting — first n rows will have NaN
# Back-shift     : converting time series into supervised learning problem
# n lags         : how many past steps to include — tune based on ACF/PACF
# Any regressor  : lag features work with LinearRegression, RF, XGBoost, etc.
