"""
K-fold cross-validation functions for model evaluation.
"""

import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score, precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd


def run_kfold(X, y, n_splits=5, random_state=None, verbose=True, include_additional_metrics=False):
    """
    Run K-Fold cross-validation using OLS (statsmodels).

    Parameters:
        X (pd.DataFrame): Feature matrix (no constant term added yet).
        y (pd.Series): Target variable.
        n_splits (int): Number of folds.
        random_state (int): Random seed for reproducibility.
        verbose (bool): Print per-fold progress.
        include_additional_metrics (bool): Include R², Adj R², AIC, BIC metrics.

    Returns:
        dict: Dictionary with fold metrics and averages
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize metrics tracking
    metrics_summary = {
        "RMSE": [],
    }

    if include_additional_metrics:
        metrics_summary.update({
            "R2": [],
            "Adj_R2": [],
            "AIC": [],
            "BIC": []
        })

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Add constant term for intercept
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)

        # Fit OLS model
        model = sm.OLS(y_train, X_train_const).fit()

        # Predict
        preds = model.predict(X_test_const)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        metrics_summary["RMSE"].append(rmse)

        if include_additional_metrics:
            metrics_summary["R2"].append(model.rsquared)
            metrics_summary["Adj_R2"].append(model.rsquared_adj)
            metrics_summary["AIC"].append(model.aic)
            metrics_summary["BIC"].append(model.bic)

        if verbose:
            if include_additional_metrics:
                print(f"Fold {fold} | R²: {model.rsquared:.4f} | Adj R²: {model.rsquared_adj:.4f} | "
                      f"AIC: {model.aic:.2f} | BIC: {model.bic:.2f} | RMSE: {rmse:.4f}")
            else:
                print(f"Fold {fold} | RMSE: {rmse:.4f}")

    # Calculate averages
    results = {}
    for metric, values in metrics_summary.items():
        mean_value = np.mean(values)
        std_value = np.std(values) if metric == "RMSE" else None

        results[f"fold_{metric.lower()}"] = values
        results[f"mean_{metric.lower()}"] = mean_value

        if std_value is not None:
            results[f"std_{metric.lower()}"] = std_value

        if verbose:
            if std_value is not None:
                print(f"{metric}: Mean = {mean_value:.4f}, SD = {std_value:.4f}")
            else:
                print(f"{metric}: Mean = {mean_value:.4f}")

    return results


def kfold_with_detailed_metrics(X, y, n_splits=5, random_state=None):
    """
    Run k-fold cross-validation with comprehensive metrics output.

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        n_splits (int): Number of folds
        random_state (int): Random seed

    Returns:
        dict: Comprehensive metrics results
    """
    return run_kfold(X, y, n_splits=n_splits, random_state=random_state,
                    verbose=True, include_additional_metrics=True)


def simple_kfold_rmse(X, y, n_splits=5, random_state=None):
    """
    Run simple k-fold cross-validation returning only RMSE metrics.

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        n_splits (int): Number of folds
        random_state (int): Random seed

    Returns:
        dict: RMSE metrics only
    """
    return run_kfold(X, y, n_splits=n_splits, random_state=random_state,
                    verbose=True, include_additional_metrics=False)


def run_kfold_classification(X, y, n_splits=5, random_state=None, verbose=True,
                           scale_features=True, model_params=None):
    """
    Run K-Fold cross-validation for classification problems using Logistic Regression.

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable (binary classification)
        n_splits (int): Number of folds
        random_state (int): Random seed for reproducibility
        verbose (bool): Print per-fold progress and metrics
        scale_features (bool): Whether to scale features using StandardScaler
        model_params (dict): Additional parameters for LogisticRegression

    Returns:
        dict: Dictionary with fold metrics and averages
    """
    if model_params is None:
        model_params = {'fit_intercept': True, 'solver': 'liblinear'}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize metrics tracking
    metrics_summary = {
        "accuracy": [],
        "precision": [],
        "auc": []
    }

    scaler = StandardScaler() if scale_features else None

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale features if requested
        if scale_features:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # Train logistic regression
        model = LogisticRegression(**model_params)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob[:, 1])

        metrics_summary["accuracy"].append(accuracy)
        metrics_summary["precision"].append(precision)
        metrics_summary["auc"].append(auc)

        if verbose:
            print(f"Fold {fold} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | AUC: {auc:.4f}")

            # Show confusion matrix for each fold
            cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
            print("Confusion Matrix:")
            print(cm)
            print("-" * 50)

    # Calculate averages
    results = {}
    for metric, values in metrics_summary.items():
        mean_value = np.mean(values)
        std_value = np.std(values)

        results[f"fold_{metric}"] = values
        results[f"mean_{metric}"] = mean_value
        results[f"std_{metric}"] = std_value

        if verbose:
            print(f"{metric.upper()}: Mean = {mean_value:.4f}, SD = {std_value:.4f}")

    return results


def kfold_classification_simple(X, y, n_splits=5, random_state=None):
    """
    Run simple k-fold cross-validation for classification returning key metrics.

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable (binary)
        n_splits (int): Number of folds
        random_state (int): Random seed

    Returns:
        dict: Classification metrics (accuracy, precision, AUC)
    """
    return run_kfold_classification(X, y, n_splits=n_splits, random_state=random_state,
                                  verbose=True, scale_features=True)