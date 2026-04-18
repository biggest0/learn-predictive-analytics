# =============================================================================
# BAGGING WORKFLOW CHEATSHEET
# Bootstrap Aggregating: train many models on bootstrap samples, aggregate preds
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

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
# 2. BASELINE: Single Decision Tree (no bagging)
# -----------------------------------------------------------------------------
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
tree_acc = accuracy_score(y_test, single_tree.predict(X_test))
print(f"Single tree accuracy:  {tree_acc:.4f}")

# -----------------------------------------------------------------------------
# 3. MANUAL BAGGING (from scratch — shows the core idea)
#    Key steps: bootstrap sample → fit base learner → majority vote
# -----------------------------------------------------------------------------
n_estimators = 50
predictions = []

for i in range(n_estimators):
    # Bootstrap: sample N rows WITH REPLACEMENT
    idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot, y_boot = X_train[idx], y_train[idx]

    model = DecisionTreeClassifier(random_state=i)
    model.fit(X_boot, y_boot)
    predictions.append(model.predict(X_test))

# Majority vote across all estimators
predictions = np.array(predictions)           # shape: (n_estimators, n_samples)
majority_vote = np.apply_along_axis(
    lambda col: np.bincount(col).argmax(), axis=0, arr=predictions
)
manual_acc = accuracy_score(y_test, majority_vote)
print(f"Manual bagging accuracy: {manual_acc:.4f}")

# -----------------------------------------------------------------------------
# 4. SKLEARN BaggingClassifier (the standard way)
# -----------------------------------------------------------------------------
bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),   # base learner
    n_estimators=100,                     # number of bootstrap models
    max_samples=1.0,                      # fraction of training set per model
    max_features=1.0,                     # fraction of features per model
    bootstrap=True,                       # True = bagging, False = pasting
    bootstrap_features=False,
    oob_score=True,                       # out-of-bag estimate (free val set)
    n_jobs=-1,
    random_state=42,
)
bag_clf.fit(X_train, y_train)

bag_acc = accuracy_score(y_test, bag_clf.predict(X_test))
oob_acc = bag_clf.oob_score_             # ~= cross-val score, no extra split needed
print(f"BaggingClassifier accuracy: {bag_acc:.4f}  |  OOB score: {oob_acc:.4f}")

# -----------------------------------------------------------------------------
# 5. RANDOM FOREST (bagging + random feature subsets at each split)
#    Extra randomness → lower correlation between trees → better variance reduction
# -----------------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",   # sqrt(n_features) per split — key RF hyperparameter
    max_depth=None,        # fully grown trees (bias stays low)
    oob_score=True,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)

rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest accuracy:     {rf_acc:.4f}  |  OOB score: {rf.oob_score_:.4f}")

# Feature importances (mean decrease in impurity across all trees)
importances = pd.Series(rf.feature_importances_, index=data.feature_names)
print("\nTop 5 features:\n", importances.nlargest(5))

# -----------------------------------------------------------------------------
# 6. CROSS-VALIDATION COMPARISON
# -----------------------------------------------------------------------------
models = {
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Bagging (100 trees)": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, oob_score=False,
                                              n_jobs=-1, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                  random_state=42),
}

print("\n--- 5-Fold CV Accuracy ---")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"{name:<25} mean={scores.mean():.4f}  std={scores.std():.4f}")

# -----------------------------------------------------------------------------
# KEY CONCEPTS (exam reminders)
# -----------------------------------------------------------------------------
# Bootstrap sample   : N rows drawn WITH replacement from training set (~63% unique)
# Out-of-bag (OOB)   : ~37% of rows not seen by each tree → free validation estimate
# Bagging reduces    : VARIANCE (not bias) — helps high-variance models like deep trees
# Pasting            : same as bagging but WITHOUT replacement (bootstrap=False)
# Random Forest extra: random feature subset at EACH SPLIT, not just per tree
# max_features       : "sqrt" for classification, "log2" or 1/3 for regression
# Aggregation        : majority vote (classification) or mean (regression)







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
# 2. SKLEARN BaggingClassifier (the standard way)
# -----------------------------------------------------------------------------
bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),   # base learner
    n_estimators=100,                     # number of bootstrap models
    max_samples=1.0,                      # fraction of training set per model
    max_features=1.0,                     # fraction of features per model
    bootstrap=True,                       # True = bagging, False = pasting
    bootstrap_features=False,
    oob_score=True,                       # out-of-bag estimate (free val set)
    n_jobs=-1,
    random_state=42,
)
bag_clf.fit(X_train, y_train)

bag_preds2 = bag_clf.predict(X_test)
bag_acc = accuracy_score(y_test, bag_preds2)
oob_acc = bag_clf.oob_score_             # ~= cross-val score, no extra split needed
print(f"BaggingClassifier accuracy: {bag_acc:.4f}  |  OOB score: {oob_acc:.4f}")
print(f"Precision (macro): {precision_score(y_test, bag_preds2, average='macro'):.4f}")
print(f"Recall    (macro): {recall_score(y_test, bag_preds2, average='macro'):.4f}")
print(f"F1        (macro): {f1_score(y_test, bag_preds2, average='macro'):.4f}")
print(classification_report(y_test, bag_preds2, target_names=data.target_names))