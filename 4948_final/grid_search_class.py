import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Load data
X, y =
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# Base model (keep your key settings)
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
param_grid = {
    "max_depth": [3, 4, 5], # model capacity / regularization
    "min_child_weight": [1, 3, 5],
    "learning_rate": [0.01, 0.05, 0.1], # boosting behavior
    "n_estimators": [200, 400, 800],
    "subsample": [0.7, 0.8, 1.0], # randomness (overfitting control)
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma": [0, 0.5, 1.0],  # split/pruning regularization
    "reg_lambda": [1, 5, 10], # weight regularization
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=xgb_clf, param_grid=param_grid,
    scoring={"AUC": "roc_auc", "LogLoss": "neg_log_loss", "Accuracy": "accuracy"},
    refit="AUC", cv=cv, n_jobs=-1, verbose=3
)
grid.fit(X_train, y_train)
print("Best CV AUC:", grid.best_score_)
print("Best params:", grid.best_params_)
best_model = grid.best_estimator_
# Evaluate on test set
preds = best_model.predict(X_test)  # already 0/1 labels
print(classification_report(y_test, preds))
