"""
Training pipeline for AirBNB price prediction model.
Uses modular functions from across the codebase.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Import from existing modules
from util.file_handler import get_csv_dataframe
from models.evaluation import save_model_and_summary, summarize_model
from feature_creation.impute import knn_imputer
from feature_creation.dummy import create_dummy_cols, create_dummy_cols_datetime
from feature_creation.bin import create_bin_cols
from feature_interpretation.k_fold import run_kfold


# Data cleaning pipeline
def clean_data():
    """
    Complete data cleaning pipeline for AirBNB data.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for modeling
    """
    df = get_csv_dataframe()

    # Impute missing values
    df = knn_imputer(df)

    # Create dummy variables for categorical data
    cols_to_dummy = ['city', 'bed_type', 'room_type', 'cancellation_policy', 'cleaning_fee']
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


def create_test_split(df, features=None, target='price', test_size=0.15, random_state=0):
    """
    Create train/test split for modeling.

    Parameters:
        df: DataFrame with features and target
        features: List of feature column names (optional, uses predefined if None)
        target: Target column name
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if features is None:
        # Default features for AirBNB model
        features = ['cancellation_policy_super_strict_60', 'bathrooms', 'smartlock',
                    'translation_missing:_en_hosting_amenity_49', 'indoor_fireplace', 'city_DC', 'doorman',
                    'room_type_Private room', 'city_NYC', 'cancellation_policy_strict', 'cable_tv', 'dryer',
                    'suitable_for_events', 'tv', 'family_kid_friendly', 'beds', 'bed_type_Couch',
                    'accommodates', 'city_SF', 'room_type_Shared room', 'city_Chicago', 'hot_tub',
                    'elevator', 'bedrooms']

    X = df[features]
    X = sm.add_constant(X)
    y = df[target]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def main():
    """
    Main training pipeline for AirBNB price prediction model.
    """
    # Clean and preprocess data
    df = clean_data()

    # Create train/test split
    X_train, y_train, X_test, y_test = create_test_split(df)

    # Perform k-fold cross validation on training data
    print("Performing k-fold cross validation...")
    kfold_results = run_kfold(X_train, y_train, n_splits=10, verbose=True, include_additional_metrics=True)

    # Build final model with all training data
    final_model = sm.OLS(y_train, X_train).fit()

    # Evaluate on held-out test data
    print("\nEvaluating on test data...")
    rmse = summarize_model(final_model, X_test, y_test)

    # Save model and summary
    save_model_and_summary(final_model, rmse)


if __name__ == '__main__':
    main()