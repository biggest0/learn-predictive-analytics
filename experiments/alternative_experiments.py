"""
Alternative testing and experimentation with modeling approaches.
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import statsmodels.api as sm
import numpy as np

from util.file_handler import get_csv_dataframe, save_local_model, load_local_model
from feature_creation.dummy import create_dummy_cols, create_dummy_cols_datetime
from feature_creation.bin import create_bin_cols
from feature_creation.outlier_treatment import clip_outliers
from models.training import clean_data_pipeline, create_train_test_split
from models.evaluation import save_model_and_summary
from feature_interpretation.k_fold import run_kfold


def run_kfold_local(X, y, n_splits=5, random_state=42, verbose=True):
    """
    Run K-Fold cross-validation using OLS (local implementation for compatibility).

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
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
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


def run_experiment():
    """
    Run main modeling experiment with feature selection and evaluation.
    """
    # Load and prepare data
    df = get_csv_dataframe('cleaned_AirBNB_data.csv')

    # Define features based on feature selection results
    features_one = ['cancellation_policy_super_strict_60', 'neighbourhood_Bel Air/Beverly Crest', 'neighbourhood_Bellevue', 'neighbourhood_Chevy Chase, MD', 'neighbourhood_Cow Hollow', 'neighbourhood_Fort Totten', 'neighbourhood_Foxhall', 'neighbourhood_Gallaudet', 'neighbourhood_Gateway', 'neighbourhood_Judiciary Square', 'neighbourhood_Laurel Canyon', 'neighbourhood_Malibu', 'neighbourhood_Pacific Palisades', 'neighbourhood_Presidio Heights', 'neighbourhood_Sea Cliff', 'neighbourhood_Skyland', 'neighbourhood_Tribeca', 'neighbourhood_Van Nest', 'neighbourhood_West Athens', 'neighbourhood_Wilmington']
    features_two = ['accommodates', 'bedrooms', 'bathrooms', 'beds', 'room_type_Private room', 'family_kid_friendly', 'indoor_fireplace', 'tv', 'cable_tv', 'translation_missing:_en_hosting_amenity_49', 'dryer', 'washer', 'suitable_for_events', 'city_SF', 'neighbourhood_Malibu', 'room_type_Shared room', 'city_DC', 'cancellation_policy_strict', 'city_NYC', 'hot_tub']

    # Combine features (union of both sets)
    features_forward = set(features_two)
    features_recursive = set(features_one)
    features = list(features_forward.union(features_recursive))

    # Remove unwanted features
    features_to_remove = ['neighbourhood_Gateway', 'dryer', 'neighbourhood_Foxhall', 'neighbourhood_Fort Totten', 'neighbourhood_West Athens']
    for feature in features_to_remove:
        if feature in features:
            features.remove(feature)

    # Prepare data for modeling
    X = df[features]
    X = sm.add_constant(X)
    y = df['price']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    # Train model
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)

    # Print results
    print(model.summary())
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error:', rmse)

    # Run cross-validation
    run_kfold(X_train, y_train, 10)


def main():
    """
    Main function to run experiments.
    """
    run_experiment()


if __name__ == '__main__':
    main()