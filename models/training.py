"""
Model training functions for OLS and other regression models.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import numpy as np
from datetime import datetime

from util.file_handler import get_dataframe_with_features, get_csv_dataframe
from constant import TOP_FEATURES, MAYBE_FEATURES
from feature_creation.impute import convertNAcellsToNum
from create_dummy import create_dummy
from feature_creation.dummy import create_dummy_cols, create_dummy_cols_datetime
from feature_creation.bin import create_bin_cols
from feature_creation.outlier_treatment import clip_outliers
from util.file_handler import get_df_imputed, save_local_model, load_local_model
from models.evaluation import save_model_and_summary
from feature_interpretation.k_fold import run_kfold


def ols_basic():
    """
    Basic OLS model training function.

    Returns:
        None (prints results)
    """
    df = get_csv_dataframe()
    # df = create_dummy(df, 'cancellation_policy')
    print(df.head(10))
    # imputing missing data with mean of column
    df = convertNAcellsToNum('bathrooms', df, "mean")
    df = convertNAcellsToNum('bedrooms', df, "mean")
    df = convertNAcellsToNum('beds', df, "mean")

    # select features to train model
    features = ['accommodates', 'imp_bedrooms', 'imp_bathrooms', 'imp_beds']
    test_features = ['dryer', 'family_kid_friendly', 'indoor_fireplace', 'tv']
    test_features.remove('family_kid_friendly')
    # dryer, family_kid_friendly
    features = features + test_features
    X = df[features]

    print(X.head(10))

    X = sm.add_constant(X)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model

    print(model.summary())
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


def train_ols_with_feature_selection(features, df=None, test_size=0.2, random_state=0, run_kfold_cv=True, n_splits=10):
    """
    Train OLS model with selected features and optional cross-validation.

    Parameters:
        features: List of feature names to use
        df: DataFrame to use (if None, loads from saved model)
        test_size: Test set proportion
        random_state: Random seed
        run_kfold_cv: Whether to run k-fold cross validation
        n_splits: Number of k-fold splits

    Returns:
        tuple: (model, rmse, predictions)
    """
    if df is None:
        df = load_local_model('test_model.pkl')

    df = clip_outliers(df)
    print(df.head(10))

    X = df[features]
    X = sm.add_constant(X)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)

    print(model.summary())
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error:', rmse)

    if run_kfold_cv:
        run_kfold(X, y, n_splits)

    # Save model and summary
    save_model_and_summary(model, rmse)

    str_feature = ', '.join(features)
    now = datetime.now().time()

    with open(f"data/ols_summary/ols_{now}_{rmse:.2f}.txt", "w") as f:
        f.write(model.summary().as_text())
        f.write(f'\nSelected features: {str_feature}')
        f.write(f'\nRoot Mean Squared Error:, {np.sqrt(metrics.mean_squared_error(y_test, predictions))}')

    return model, rmse, predictions


def prepare_data_for_modeling():
    """
    Prepare data for modeling by applying all preprocessing steps.

    Returns:
        pd.DataFrame: Processed DataFrame ready for modeling
    """
    df = get_df_imputed()

    # Create dummy variables for categorical columns
    from constant import COLUMNS_TO_DUMMY
    for dummy in COLUMNS_TO_DUMMY:
        df = create_dummy_cols(df, dummy)

    # Create dummy variables for datetime column
    df = create_dummy_cols_datetime(df, 'host_since')

    # Create bins for review scores
    df = create_bin_cols(df, 'review_scores_rating', [0, 20, 40, 60, 80, 100])

    # Save processed data
    save_local_model(df, 'test')

    # Select only numeric columns for modeling
    df_numeric = df.select_dtypes(include='number')

    return df_numeric


def clean_data_pipeline(csv_path='data/AirBNB.csv'):
    """
    Complete data cleaning pipeline for AirBNB data.

    Parameters:
        csv_path: Path to the CSV file

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df = get_csv_dataframe(csv_path)

    # Impute missing values using KNN
    from assignment1.train import knn_imputer
    df = knn_imputer(df)

    # Create dummy variables
    cols_to_dummy = ['city', 'bed_type', 'room_type', 'cancellation_policy', 'cleaning_fee', 'neighbourhood']
    for col in cols_to_dummy:
        df = create_dummy_cols(df, col)

    # Create dummy variables for datetime
    df = create_dummy_cols_datetime(df, 'host_since')

    # Create bins for review scores
    df = create_bin_cols(df, 'review_scores_rating', [0, 20, 40, 60, 80, 100])

    # Save cleaned data
    output_path = "cleaned_AirBNB_data.csv"
    df.to_csv(output_path, index=False)

    return df


def create_train_test_split(df, features, target='price', test_size=0.15, random_state=0):
    """
    Create train/test split for modeling.

    Parameters:
        df: DataFrame with features and target
        features: List of feature column names
        target: Target column name
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df[features]
    X = sm.add_constant(X)
    y = df[target]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
