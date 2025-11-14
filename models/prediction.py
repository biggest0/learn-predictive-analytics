"""
Prediction and data preparation functions for making predictions on new data.
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np

from util.file_handler import get_csv_dataframe
from feature_creation.dummy import create_dummy_cols, create_dummy_cols_datetime
from feature_creation.bin import create_bin_cols
from util.file_handler import load_local_model
from constant import COLUMNS_TO_DUMMY


def make_missing_cols(df_train_features, df_prediction):
    """
    Add missing columns to prediction DataFrame to match training features.

    Parameters:
        df_train_features: List of feature names from training data
        df_prediction: Prediction DataFrame to modify

    Returns:
        pd.DataFrame: Modified prediction DataFrame with missing columns added
    """
    train_test_columns = df_train_features
    print("Training features:", train_test_columns)

    prediction_columns = list(df_prediction.keys())
    print("Prediction features:", prediction_columns)

    for i in range(0, len(train_test_columns)):
        column_found = False
        for j in range(0, len(prediction_columns)):
            if train_test_columns[i] == prediction_columns[j]:
                column_found = True
                break
        # Add column and store zeros in every cell if not found.
        if not column_found:
            col_name = train_test_columns[i]
            df_prediction[col_name] = 0

    return df_prediction


def predict_mystery_data(model_path='data/pickle_model/model_23:40:24.621383_126.16.pkl',
                        mystery_path='./data/AirBNB_mystery.csv'):
    """
    Make predictions on mystery data using a trained model.

    Parameters:
        model_path: Path to saved model pickle file
        mystery_path: Path to mystery data CSV

    Returns:
        pd.Series: Predictions for mystery data
    """
    # Load trained model
    model = load_local_model(model_path)

    # Load mystery data
    mystery = get_csv_dataframe(mystery_path)

    # Apply same preprocessing as training data
    for dummy in COLUMNS_TO_DUMMY:
        mystery = create_dummy_cols(mystery, dummy)

    mystery = create_dummy_cols_datetime(mystery, 'host_since')
    mystery = create_bin_cols(mystery, 'review_scores_rating', [0, 20, 40, 60, 80, 100])

    # Ensure mystery data has all required features
    make_missing_cols(model.model.exog_names, mystery)

    print("Mystery data after preprocessing:")
    print(mystery.head(10))

    # Get feature columns (excluding constant)
    feature_cols = [col for col in model.model.exog_names if col != 'const']
    print("Features used in model:", feature_cols)

    # Prepare data for prediction
    X_mystery = mystery[feature_cols].copy()
    X_mystery = sm.add_constant(X_mystery)

    # Make predictions
    predictions = model.predict(X_mystery)

    print("Predictions for mystery data:")
    print(predictions.head(10))

    return predictions


def prepare_data_for_prediction(df, feature_columns):
    """
    Prepare new data for prediction by ensuring it has all required features.

    Parameters:
        df: DataFrame with new data
        feature_columns: List of required feature column names

    Returns:
        pd.DataFrame: Prepared DataFrame with constant term added
    """
    # Ensure all required features exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Add constant term for statsmodels
    df_prepared = sm.add_constant(df[feature_columns])

    return df_prepared


def batch_predict(model, X_new, feature_names=None):
    """
    Make batch predictions using a trained model.

    Parameters:
        model: Trained statsmodels model
        X_new: New data for prediction (DataFrame or array-like)
        feature_names: List of feature names if X_new is array-like

    Returns:
        np.array: Predictions
    """
    if isinstance(X_new, pd.DataFrame):
        # Ensure DataFrame has constant term
        if 'const' not in X_new.columns:
            X_new = sm.add_constant(X_new)
        predictions = model.predict(X_new)
    else:
        # Convert array-like to DataFrame with proper column names
        if feature_names is None:
            raise ValueError("feature_names must be provided when X_new is array-like")

        X_df = pd.DataFrame(X_new, columns=feature_names)
        X_df = sm.add_constant(X_df)
        predictions = model.predict(X_df)

    return predictions


def predict_with_confidence_intervals(model, X_new, alpha=0.05):
    """
    Make predictions with confidence intervals.

    Parameters:
        model: Trained statsmodels model
        X_new: New data for prediction
        alpha: Significance level for confidence intervals

    Returns:
        tuple: (predictions, confidence_intervals)
    """
    if isinstance(X_new, pd.DataFrame) and 'const' not in X_new.columns:
        X_new = sm.add_constant(X_new)

    predictions = model.predict(X_new)

    # Get prediction confidence intervals
    pred_results = model.get_prediction(X_new)
    conf_int = pred_results.conf_int(alpha=alpha)

    return predictions, conf_int
