# =============================================================================
# EARLY STOPPING CHEATSHEET (skorch + PyTorch)
#   Class-style Net  →  RandomizedSearchCV  →  EarlyStopping callback
# =============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, EpochScoring

# -----------------------------------------------------------------------------
# 1. LOAD & SPLIT DATA
# -----------------------------------------------------------------------------
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop('target', axis=1).values.astype(np.float32)
y = df['target'].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# -----------------------------------------------------------------------------
# 2. DEFINE Net (nn.Module)  — matches class-slide style
# -----------------------------------------------------------------------------
class Net(nn.Module):
    NUM_FEATURES = X_train.shape[1]       # 13 features in wine dataset
    OUTPUT_DIM   = len(np.unique(y))      # 3 classes

    def __init__(self, num_neurons=16):
        super(Net, self).__init__()
        # 1st hidden layer
        self.dense0         = nn.Linear(self.NUM_FEATURES, num_neurons)
        self.activationFunc = nn.ReLU()
        # Drop samples to help prevent overfitting
        DROPOUT      = 0.1
        self.dropout = nn.Dropout(DROPOUT)
        # 2nd hidden layer
        self.dense1  = nn.Linear(num_neurons, self.OUTPUT_DIM)
        # Output layer
        self.output  = nn.Linear(self.OUTPUT_DIM, self.OUTPUT_DIM)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.to(torch.float32)
        X = self.activationFunc(self.dense0(x))
        X = self.dropout(X)
        X = self.activationFunc(self.dense1(X))
        X = self.softmax(self.output(X))
        return X

# -----------------------------------------------------------------------------
# 3. CALLBACKS  — EarlyStopping halts training when loss stops improving
#    EpochScoring : logs accuracy each epoch (on_train=True → training set)
#    EarlyStopping: monitor='train_loss' because train_split=False below
# -----------------------------------------------------------------------------
callbacks = [
    EpochScoring(scoring='accuracy', name='train_acc', on_train=True),
    EarlyStopping(monitor='train_loss', patience=100),   # stop if no improvement for 100 epochs
]

# -----------------------------------------------------------------------------
# 4. WRAP WITH SKORCH NeuralNetClassifier
# -----------------------------------------------------------------------------
nn_clf = NeuralNetClassifier(
    Net,
    max_epochs=200,
    lr=0.001,
    batch_size=100,
    optimizer=optim.RMSprop,
    callbacks=callbacks,
    train_split=False,     # outer RandomizedSearchCV already does CV
    verbose=0,
)

# -----------------------------------------------------------------------------
# 5. RANDOMIZED SEARCH CV  — try combos of hyperparameters
# -----------------------------------------------------------------------------
params = {
    'max_epochs':          [50, 100, 150],
    'lr':                  [0.1, 0.007, 0.005, 0.001],
    'module__num_neurons': [5, 10, 30],                      # constructor arg of Net
    'optimizer':           [optim.Adam, optim.SGD, optim.RMSprop],
}

gs = RandomizedSearchCV(
    nn_clf, params, refit=True, cv=3,
    scoring='balanced_accuracy', verbose=1,
)
gs.fit(X_train, y_train)

print("\nBest parameters:")
print(gs.best_params_)

# -----------------------------------------------------------------------------
# 6. EVALUATE BEST MODEL
# -----------------------------------------------------------------------------
def evaluateModel(model, X_test, y_test):
    print(model)
    y_pred = model.predict(X_test)
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print("Confusion matrix")
    print(cm)
    print(classification_report(y_test, y_pred, target_names=data.target_names))

evaluateModel(gs.best_estimator_, X_test, y_test)

# -----------------------------------------------------------------------------
# 7. SAVE BEST MODEL AS PICKLE
# -----------------------------------------------------------------------------
import pickle

with open('best_model.pkl', 'wb') as f:
    pickle.dump(gs.best_estimator_, f)
print("\nSaved best model → best_model.pkl")

# Load it back (sanity check)
with open('best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
print("Loaded model accuracy:",
      (loaded_model.predict(X_test) == y_test).mean())

# -----------------------------------------------------------------------------
# KEY CONCEPTS (exam reminders)
# -----------------------------------------------------------------------------
# EarlyStopping      : stops training when monitored metric stops improving
# patience           : epochs to wait for improvement before stopping
# monitor            : which metric to track — 'valid_loss' (default) or 'train_loss'
# EpochScoring       : logs a metric each epoch (on_train=True → training set)
# NeuralNetClassifier: skorch wrapper — gives nn.Module sklearn fit/predict API
# train_split=False  : disables skorch's internal val split — use when outer CV exists
# RandomizedSearchCV : samples random combos from param grid (faster than full grid)
# module__<arg>      : passes <arg> into Net's __init__ (e.g. module__num_neurons)
