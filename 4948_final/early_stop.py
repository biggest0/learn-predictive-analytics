# =============================================================================
# EARLY STOPPING CHEATSHEET (skorch + PyTorch)
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, EpochScoring

# -----------------------------------------------------------------------------
# 1. LOAD & SPLIT DATA
# -----------------------------------------------------------------------------
data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)
y_train = y_train.astype(np.int64)
y_test  = y_test.astype(np.int64)

# -----------------------------------------------------------------------------
# 2. DEFINE ANN (nn.Module)
# -----------------------------------------------------------------------------
class ANN(nn.Module):
    def __init__(self, input_size=13, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# 3. DEFINE CALLBACKS
#    EpochScoring : logs accuracy each epoch (on_train=True → training set)
#    EarlyStopping: stops training when val accuracy stops improving
# -----------------------------------------------------------------------------
callbacks = [
    EpochScoring(scoring='accuracy', name='train_acc', on_train=True),
    EarlyStopping(patience=100),    # stop if no improvement for 100 epochs
]

# -----------------------------------------------------------------------------
# 4. WRAP WITH SKORCH NeuralNetClassifier
# -----------------------------------------------------------------------------
model = NeuralNetClassifier(
    module=ANN,
    module__input_size=X_train.shape[1],
    module__num_classes=len(np.unique(y)),
    max_epochs=500,
    lr=0.01,
    optimizer=torch.optim.Adam,
    criterion=nn.CrossEntropyLoss,
    callbacks=callbacks,
    verbose=1,
)

# -----------------------------------------------------------------------------
# 5. TRAIN
# -----------------------------------------------------------------------------
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 6. EVALUATE
# -----------------------------------------------------------------------------
preds = model.predict(X_test)
print(f"\nAccuracy : {accuracy_score(y_test, preds):.4f}")
print(f"Precision: {precision_score(y_test, preds, average='macro'):.4f}")
print(f"Recall   : {recall_score(y_test, preds, average='macro'):.4f}")
print(f"F1       : {f1_score(y_test, preds, average='macro'):.4f}")
print(classification_report(y_test, preds, target_names=data.target_names))

# -----------------------------------------------------------------------------
# KEY CONCEPTS (exam reminders)
# -----------------------------------------------------------------------------
# EarlyStopping  : stops training when monitored metric stops improving
# patience       : how many epochs to wait before stopping (default monitors val loss)
# EpochScoring   : logs a metric each epoch — on_train=True logs on training set
# NeuralNetClassifier : skorch wrapper — gives nn.Module sklearn fit/predict API
# max_epochs     : upper bound; early stopping will halt before this if needed
# astype float32 : skorch/PyTorch requires float32 for X, int64 for y
