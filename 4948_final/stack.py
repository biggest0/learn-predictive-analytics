# =============================================================================
# STACKING WORKFLOW CHEATSHEET
# Train base models → use their predictions as features → fit a meta-model
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
data = load_wine()
X, y = data.data, data.target

# -----------------------------------------------------------------------------
# 2. SPLIT INTO TRAIN / VAL / TEST  (stacking needs 3 sets)
#    train → fit base models
#    val   → base model predictions become meta-features for the meta-model
#    test  → final evaluation
# -----------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_val,   X_test,  y_val,  y_test  = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# 3. DEFINE & FIT BASE MODELS ON TRAIN SET
# -----------------------------------------------------------------------------
base_models = [
    DecisionTreeClassifier(random_state=42),
    SVC(probability=True, random_state=42),
    AdaBoostClassifier(random_state=42),
    RandomForestClassifier(n_estimators=100, random_state=42),
    ExtraTreesClassifier(n_estimators=100, random_state=42),
]

for model in base_models:
    model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 4. BUILD META-FEATURES — base model predictions on VAL set
# -----------------------------------------------------------------------------
val_meta = pd.DataFrame()
for i in range(len(base_models)):
    val_meta[str(i)] = base_models[i].predict(X_val)

# -----------------------------------------------------------------------------
# 5. FIT META-MODEL (stacked model) ON VAL META-FEATURES
# -----------------------------------------------------------------------------
meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(val_meta, y_val)

# -----------------------------------------------------------------------------
# 6. PREDICT ON TEST SET — base models first, then meta-model
# -----------------------------------------------------------------------------
test_meta = pd.DataFrame()
for i in range(len(base_models)):
    test_meta[str(i)] = base_models[i].predict(X_test)

final_preds = meta_model.predict(test_meta)

# -----------------------------------------------------------------------------
# 7. EVALUATE STACKED MODEL
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
# ExtraTrees vs RF   : ExtraTrees splits randomly at each node → faster, more variance
