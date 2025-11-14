import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def run_kfold(X, y, n_splits=5, random_state=42, verbose=True):
    """
    Run K-Fold cross-validation using OLS (statsmodels).

    Parameters:
        X (pd.DataFrame): Feature matrix (no constant term added yet).
        y (pd.Series): Target variable.
        n_splits (int): Number of folds.
        random_state (int): Random seed.
        verbose (bool): Print per-fold progress.

    Returns:
        dict: {
            "fold_rmse": list of RMSE values for each fold,
            "mean_rmse": average RMSE across folds
        }
    """
    kf = KFold(n_splits=n_splits, shuffle=True)
    rmse_scores = []

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
        rmse_scores.append(rmse)

        if verbose:
            print(f"Fold {fold} | RMSE: {rmse:.4f}")

    # Summary
    mean_rmse = np.mean(rmse_scores)
    if verbose:
        print(f"\nAverage RMSE across {n_splits} folds: {mean_rmse:.4f}")

    return {
        "fold_rmse": rmse_scores,
        "mean_rmse": mean_rmse
    }


def class_kfold(X, y, NUM_SPLITS=5):
    cv = KFold(n_splits=NUM_SPLITS, shuffle=True)
    rmse_scores = []
    # for train_indices, test_indicies in cv.split(X):
    for n, (train_indices, test_indicies) in enumerate(cv.split(X), start=1):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indicies]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indicies]

        # Add constant term for intercept
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)

        # Fit OLS model
        model = sm.OLS(y_train, X_train_const).fit()

        # Predict
        preds = model.predict(X_test_const)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        rmse_scores.append(rmse)
        print(f"Fold {n} | RMSE: {rmse:.4f}")

    mean_rmse = np.mean(rmse_scores)
    print(f"\nAverage RMSE across {NUM_SPLITS} folds: {mean_rmse:.4f}")