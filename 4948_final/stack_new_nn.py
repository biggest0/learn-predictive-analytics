# =============================================================================
# STACKING WORKFLOW CHEATSHEET  (with skorch Net)
# Train base models → use their predictions as features → fit a meta-model
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from skorch import NeuralNetClassifier

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop('target', axis=1).values.astype(np.float32)
y = df['target'].values.astype(np.int64)

# -----------------------------------------------------------------------------
# 2. SPLIT INTO TRAIN / VAL / TEST  (stacking needs 3 sets)
#    train → fit base models
#    val   → base model predictions become meta-features for the meta-model
#    test  → final evaluation
# -----------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_val,   X_test,  y_val,  y_test  = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# -----------------------------------------------------------------------------
# 3a. DEFINE Net (nn.Module) — class-slide style
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
# 3b. WRAP Net WITH SKORCH — gives sklearn-style fit/predict
# -----------------------------------------------------------------------------
ann = NeuralNetClassifier(
    Net,
    max_epochs=200,
    lr=0.001,
    batch_size=100,
    optimizer=optim.RMSprop,
    train_split=False,
    verbose=0,
)

# -----------------------------------------------------------------------------
# 4. DEFINE & FIT BASE MODELS ON TRAIN SET
# -----------------------------------------------------------------------------
base_models = [
    DecisionTreeClassifier(random_state=42),
    SVC(probability=True, random_state=42),
    AdaBoostClassifier(random_state=42),
    RandomForestClassifier(n_estimators=100, random_state=42),
    ExtraTreesClassifier(n_estimators=100, random_state=42),
    ann,   # skorch NeuralNetClassifier wrapping Net
]

for model in base_models:
    model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 5. BUILD META-FEATURES — base model predictions on VAL set
# -----------------------------------------------------------------------------
val_meta = pd.DataFrame()
for i in range(len(base_models)):
    val_meta[str(i)] = base_models[i].predict(X_val)

# -----------------------------------------------------------------------------
# 6. FIT META-MODEL (stacked model) ON VAL META-FEATURES
# -----------------------------------------------------------------------------
meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(val_meta, y_val)

# -----------------------------------------------------------------------------
# 7. PREDICT ON TEST SET — base models first, then meta-model
# -----------------------------------------------------------------------------
test_meta = pd.DataFrame()
for i in range(len(base_models)):
    test_meta[str(i)] = base_models[i].predict(X_test)

final_preds = meta_model.predict(test_meta)

# -----------------------------------------------------------------------------
# 8. EVALUATE STACKED MODEL
# -----------------------------------------------------------------------------
print(f"Accuracy : {accuracy_score(y_test, final_preds):.4f}")
print(f"Precision: {precision_score(y_test, final_preds, average='macro'):.4f}")
print(f"Recall   : {recall_score(y_test, final_preds, average='macro'):.4f}")
print(f"F1       : {f1_score(y_test, final_preds, average='macro'):.4f}")
print(classification_report(y_test, final_preds, target_names=data.target_names))

# -----------------------------------------------------------------------------
# KEY CONCEPTS (exam reminders)
# -----------------------------------------------------------------------------
# 3 splits required  : train (base) → val (meta-features) → test (final eval)
# Meta-features      : base model predictions on val set → columns fed to meta-model
# Meta-model         : learns HOW to combine base predictions (LogReg here)
# No data leakage    : base models never see val; meta-model never sees train
# skorch wrapper     : lets PyTorch nn.Module plug into sklearn-style fit/predict
# train_split=False  : skip skorch's internal val split (train on all given data)
