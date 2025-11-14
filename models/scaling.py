"""
Scaling functions for model training and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from pickle import dump, load


def scaling_OLS():
    """
    Example of OLS regression with RobustScaler for both features and target.
    Uses wine dataset for demonstration.
    """
    wine = datasets.load_wine()
    dataset = pd.DataFrame(
        data=np.c_[wine['data'], wine['target']],
        columns=wine['feature_names'] + ['target']
    )

    # Create copy to prevent overwrite.
    X = dataset.copy()
    del X['target']  # Remove target variable
    del X['hue']  # Remove unwanted features
    del X['ash']
    del X['magnesium']
    del X['malic_acid']
    del X['alcohol']

    y = dataset['target']

    # Adding an intercept *** This is requried ***. Don't forget this step.
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    from sklearn.preprocessing import RobustScaler

    sc_x = RobustScaler()
    X_train_scaled = sc_x.fit_transform(X_train)

    # Create y scaler. Only scale y_train since evaluation
    # will use the actual size y_test.
    sc_y = RobustScaler()
    y_train_scaled = sc_y.fit_transform(np.array(y_train).reshape(-1, 1))

    # Save the fitted scalers.
    dump(sc_x, open('sc_x.pkl', 'wb'))
    dump(sc_y, open('sc_y.pkl', 'wb'))

    # Build model with training data.
    model = sm.OLS(y_train_scaled, X_train_scaled).fit()

    # Save model
    dump(model, open('ols_model.pkl', 'wb'))

    # Load model
    loaded_model = load(open('ols_model.pkl', 'rb'))

    # Load the scalers.
    loaded_scalerX = load(open('sc_x.pkl', 'rb'))
    loaded_scalery = load(open('sc_y.pkl', 'rb'))

    X_test_scaled = loaded_scalerX.transform(X_test)
    scaledPredictions = loaded_model.predict(X_test_scaled)  # make predictions

    # Rescale predictions back to actual size range.
    predictions = loaded_scalery.inverse_transform(
        np.array(scaledPredictions).reshape(-1, 1))

    print(loaded_model.summary())
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, predictions)))


def scale_and_train_ols(X_train, y_train, X_test, y_test, scaler_type='robust', save_scalers=False):
    """
    Scale features and target, train OLS model, and return predictions.

    Parameters:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        scaler_type: Type of scaler ('robust' or 'standard')
        save_scalers: Whether to save scalers to pickle files

    Returns:
        tuple: (model, predictions, rmse)
    """
    from sklearn.preprocessing import StandardScaler

    # Choose scaler
    if scaler_type == 'robust':
        scaler_X = RobustScaler()
        scaler_y = RobustScaler()
    elif scaler_type == 'standard':
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
    else:
        raise ValueError("scaler_type must be 'robust' or 'standard'")

    # Scale features
    X_train_scaled = scaler_X.fit_transform(X_train)

    # Scale target (only training)
    y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1))

    # Save scalers if requested
    if save_scalers:
        dump(scaler_X, open('scaler_X.pkl', 'wb'))
        dump(scaler_y, open('scaler_y.pkl', 'wb'))

    # Train model
    model = sm.OLS(y_train_scaled, X_train_scaled).fit()

    # Make predictions on scaled test data
    X_test_scaled = scaler_X.transform(X_test)
    scaled_predictions = model.predict(X_test_scaled)

    # Rescale predictions back to original scale
    predictions = scaler_y.inverse_transform(
        np.array(scaled_predictions).reshape(-1, 1)).flatten()

    # Calculate RMSE
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

    return model, predictions, rmse


def scaling_logistic():
    """
    Example of logistic regression with scaling using synthetic data.
    """
    # --- Example dataset ---
    df = pd.DataFrame({
        'income': [25, 30, 35, 40, 100, 120, 150, 200],
        'age': [20, 25, 30, 35, 40, 45, 50, 55],
        'bought': [0, 0, 0, 1, 1, 1, 1, 1]
    })

    # Features and target
    X = df[['income', 'age']]
    y = df['bought']

    # --- 1. Scale features ---
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 2. Split data ---
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

    # --- 3. Train logistic regression ---
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)

    # --- 4. Predict and evaluate ---
    y_pred = model.predict(X_test)

    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))


def scale_and_train_logistic(X_train, y_train, X_test, y_test, scaler_type='robust', **model_kwargs):
    """
    Scale features and train logistic regression model.

    Parameters:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        scaler_type: Type of scaler ('robust' or 'standard')
        **model_kwargs: Additional arguments for LogisticRegression

    Returns:
        tuple: (model, predictions, accuracy, confusion_matrix)
    """
    from sklearn.preprocessing import StandardScaler

    # Choose scaler
    if scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("scaler_type must be 'robust' or 'standard'")

    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(**model_kwargs)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, y_pred, acc, cm


def load_scalers_and_predict(X_test, model_path='ols_model.pkl', scaler_x_path='scaler_X.pkl', scaler_y_path='scaler_y.pkl'):
    """
    Load saved scalers and model to make predictions.

    Parameters:
        X_test: Test features
        model_path: Path to saved model
        scaler_x_path: Path to saved X scaler
        scaler_y_path: Path to saved y scaler

    Returns:
        predictions: Predictions in original scale
    """
    # Load model and scalers
    model = load(open(model_path, 'rb'))
    scaler_X = load(open(scaler_x_path, 'rb'))
    scaler_y = load(open(scaler_y_path, 'rb'))

    # Scale test data
    X_test_scaled = scaler_X.transform(X_test)

    # Make predictions
    scaled_predictions = model.predict(X_test_scaled)

    # Rescale predictions back to original scale
    predictions = scaler_y.inverse_transform(
        np.array(scaled_predictions).reshape(-1, 1)).flatten()

    return predictions
