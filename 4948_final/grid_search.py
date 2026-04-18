# =============================================================================
# GRID SEARCH CHEATSHEET
# Loop over hyperparameter combos, evaluate each, find best params
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# -----------------------------------------------------------------------------
# 1. LOAD & SPLIT DATA
# -----------------------------------------------------------------------------
data = load_wine()
X, y = data.data, data.target

# -----------------------------------------------------------------------------
# 2. DEFINE PARAM GRID  ← swap values here to search different combos
# -----------------------------------------------------------------------------
n_estimators_list = [50, 100, 200]
max_depth_list    = [None, 5, 10]
max_features_list = ["sqrt", "log2"]

# -----------------------------------------------------------------------------
# 3. GRID SEARCH LOOP
# -----------------------------------------------------------------------------
results = []

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        for max_features in max_features_list:

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y
            )
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                random_state=42,
            )
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))

            results.append({
                "n_estimators": n_estimators,
                "max_depth":    max_depth,
                "max_features": max_features,
                "accuracy":     round(acc, 4),
            })

# -----------------------------------------------------------------------------
# 4. SHOW RESULTS SORTED BY ACCURACY
# -----------------------------------------------------------------------------
dfResults = pd.DataFrame(results).sort_values(by="accuracy", ascending=False)
print(dfResults)

# -----------------------------------------------------------------------------
# 5. BEST MODEL — refit with best params and evaluate
# -----------------------------------------------------------------------------
best = dfResults.iloc[0]
print(f"\nBest params: n_estimators={best['n_estimators']}  "
      f"max_depth={best['max_depth']}  max_features={best['max_features']}")

best_model = RandomForestClassifier(
    n_estimators=int(best["n_estimators"]),
    max_depth=best["max_depth"] if pd.notna(best["max_depth"]) else None,
    max_features=best["max_features"],
    random_state=42,
)
best_model.fit(X_train, y_train)
print(classification_report(y_test, best_model.predict(X_test),
                             target_names=data.target_names))

# -----------------------------------------------------------------------------
# KEY CONCEPTS (exam reminders)
# -----------------------------------------------------------------------------
# Grid search        : exhaustive search over all hyperparameter combinations
# Param grid         : nested lists of values to try — one loop per parameter
# Results df         : store each combo + score, sort to find best
# Refit best model   : after search, train final model with best params on full train set
# Swap model         : replace RandomForestClassifier + its params to search any model
