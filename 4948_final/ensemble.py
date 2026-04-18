# =============================================================================
# ENSEMBLE VOTE CLASSIFIER CHEATSHEET
# =============================================================================

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from mlxtend.classifier import EnsembleVoteClassifier

# -----------------------------------------------------------------------------
# 1. LOAD & SPLIT DATA
# -----------------------------------------------------------------------------
data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# 2. HELPER
# -----------------------------------------------------------------------------
def evaluateModel(clf, X_test, title):
    print("\n*** " + title + " ***")
    predictions = clf.predict(X_test)
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, average='macro'):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, average='macro'):.4f}")
    print(f"F1       : {f1_score(y_test, predictions, average='macro'):.4f}")
    print(classification_report(y_test, predictions, target_names=data.target_names))

# -----------------------------------------------------------------------------
# 3. BASE CLASSIFIERS
# -----------------------------------------------------------------------------
ada_boost = AdaBoostClassifier(random_state=42)
grad_boost = GradientBoostingClassifier(random_state=42)
xgb_boost  = XGBClassifier(random_state=42, eval_metric='mlogloss')

classifiers = [ada_boost, grad_boost, xgb_boost]

for clf in classifiers:
    clf.fit(X_train, y_train)
    evaluateModel(clf, X_test, clf.__class__.__name__)

# -----------------------------------------------------------------------------
# 4. ENSEMBLE VOTE CLASSIFIER
# -----------------------------------------------------------------------------
eclf = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost, xgb_boost], voting='hard')
# hard = majority of model votes win
# soft = average the votes to get confidence in each class 0/1, higher one wins
eclf.fit(X_train, y_train)
evaluateModel(eclf, X_test, "EnsembleVoteClassifier (hard voting)")

# -----------------------------------------------------------------------------
# KEY CONCEPTS (exam reminders)
# -----------------------------------------------------------------------------
# EnsembleVoteClassifier : combines multiple classifiers via majority vote
# voting='hard'          : majority class label wins
# voting='soft'          : average predicted probabilities (needs predict_proba)
# AdaBoost               : sequential, reweights misclassified samples
# GradientBoosting       : sequential, fits residuals of previous model
# XGBoost                : optimised gradient boosting, regularisation built-in
