"""fits the model repeatedly, one step at a time, always predicting a value that already exists in the test set. After each prediction it adds the real value back into training before the next step. This lets you calculate RMSE and see how accurate the model actually is.  """
# =============================================================================
# ARIMA WALK-FORWARD VALIDATION CHEATSHEET
# Retrain model each step, adding one real observation at a time
# =============================================================================

import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
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

y_train = df[0:len(df) - TEST_STEPS]
y_test  = df[len(df) - TEST_STEPS:]

# -----------------------------------------------------------------------------
# 3. WALK-FORWARD VALIDATION LOOP
#    Each iteration: fit on current_train → forecast 1 step → add real value
# -----------------------------------------------------------------------------
predictions   = []
current_train = y_train.copy()

for i in range(len(y_test)):
    print("History length: " + str(len(current_train)))

    model = ARIMA(current_train, order=(1, 1, 1)).fit()   # ← swap (p,d,q) here
    yhat  = model.forecast(steps=1)
    predictions.append(yhat.iloc[0])

    # Add the real observed value to training before next iteration
    current_train.loc[len(current_train)] = y_test.iloc[i]

# -----------------------------------------------------------------------------
# 4. EVALUATE
# -----------------------------------------------------------------------------
rmse = sqrt(mean_squared_error(y_test.values, predictions))
print('Test RMSE: %.3f' % rmse)

# -----------------------------------------------------------------------------
# 5. PLOT ACTUAL vs PREDICTIONS
# -----------------------------------------------------------------------------
plt.plot(y_test.values, label='Actual',      marker='o')
plt.plot(predictions,   label='Predictions', marker='o')
plt.legend()
plt.title("Walk-Forward ARIMA Predictions")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# KEY CONCEPTS (exam reminders)
# -----------------------------------------------------------------------------
# Walk-forward  : refit model each step using all available history
# current_train : grows by 1 real observation each loop iteration
# One-step-ahead: forecast(steps=1) — predicts only the next value
# RMSE          : sqrt(mean_squared_error) — lower is better
# order=(p,d,q) : swap p/d/q based on ACF/PACF (see arima.py)
# vs static     : walk-forward is more realistic — model sees new data each step
