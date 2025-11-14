"""
Model evaluation functions for regression and classification models.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from datetime import datetime
import pickle


def no_scaling_logistic_prediction(X_train, y_train, X_test, y_test):
    """
    Perform logistic regression without scaling.

    Parameters:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        None (prints results)
    """
    # Perform logistic regression.
    model = LogisticRegression(fit_intercept=True, solver='liblinear')

    # Fit the model.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Show confusion matrix and accuracy scores.
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix")
    print(cm)


def no_scaling_linear_prediction(X_train, y_train, X_test, y_test):
    """
    Perform linear regression without scaling and print evaluation metrics.

    Parameters:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        None (prints results)
    """
    # Perform linear regression
    model = LinearRegression()

    # Fit the model.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate regression performance
    print("\nR² (Coefficient of Determination):", r2_score(y_test, y_pred))
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))

    # Optional: compare actual vs predicted values
    comparison = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    print("\nActual vs Predicted:")
    print(comparison.head())


def save_model_and_summary(model, rmse, save_dir="data/pickle_model"):
    """
    Save model and OLS summary to files.

    Parameters:
        model: Trained statsmodels OLS model
        rmse: Root mean squared error value
        save_dir: Directory to save files

    Returns:
        None
    """
    # Get all feature names used in the model
    all_features = model.params.index.tolist()

    # Remove 'const' (intercept) if you want only actual features
    features = [f for f in all_features if f != 'const']

    str_feature = ', '.join(features)
    now = datetime.now().time()

    # Ensure save directory exists
    import os
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/model_{now}_{rmse:.2f}.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(f"data/ols_summary/ols_{now}_{rmse:.2f}.txt", "w") as f:
        f.write(model.summary().as_text())
        f.write(f'\nSelected features: {str_feature}')
        f.write(f'\nRoot Mean Squared Error:, {rmse}')


def summarize_model(model, X_test, y_test):
    """
    Print model summary and calculate RMSE for test data.

    Parameters:
        model: Trained statsmodels model
        X_test: Test features
        y_test: Test target

    Returns:
        rmse: Root mean squared error
    """
    predictions = model.predict(X_test)
    print(model.summary())
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    print('Unseen Root Mean Squared Error:', rmse)
    return rmse


def evaluate_regression_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate regression model performance with common metrics.

    Parameters:
        y_true: True target values
        y_pred: Predicted target values
        model_name: Name of the model for display

    Returns:
        dict: Dictionary with evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Evaluation:")
    print(f"R² (Coefficient of Determination): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    return {
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }
