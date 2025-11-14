"""
Testing and experimentation with different modeling approaches.
"""

from feature_interpretation.interpret_features import get_top_by_col
from feature_interpretation.feature_selection import forward_feature_selection, recursive_feature_elimination
from models.training import prepare_data_for_modeling, train_ols_with_feature_selection
from models.prediction import predict_mystery_data
from util.file_handler import get_csv_dataframe
from constant import *


def feature_selection():
    """
    Run feature selection experiments.
    """
    df = prepare_data_for_modeling()

    # Run feature selection algorithms
    recursive_feature_elimination(df)

    print("Feature selection completed.")
    print(df.head(10))


def test_ols_model():
    """
    Test OLS model with selected features.
    """
    # Get feature combinations
    features_forward = set(TWO_TOP_20_FEATURE_FORWARD)
    features_recursive = set(TWO_TOP_20_FEATURE_RECURSIVE)

    # Find common features (intersection)
    features = list(features_forward.intersection(features_recursive))

    # Alternative: all unique features (union)
    # features = list(features_forward.union(features_recursive))

    # Remove unwanted features
    features_to_remove = ['roll_in_shower_with_chair', 'washer', 'air_purifier']
    features_to_remove += ['beachfront', 'path_to_entrance_lit_at_night']
    features_to_remove += ['free_parking_on_street', 'paid_parking_off_premises']

    for feature in features_to_remove:
        if feature in features:
            features.remove(feature)

    # Train and evaluate model
    model, rmse, predictions = train_ols_with_feature_selection(
        features=features,
        run_kfold_cv=True,
        n_splits=10
    )


def test_mystery_data():
    """
    Test predictions on mystery data.
    """
    predictions = predict_mystery_data()


def exploratory_analysis():
    """
    Basic exploratory data analysis.
    """
    df = get_csv_dataframe()
    prices = df['price']
    count_above_500 = (prices > 500).sum()
    print(f"Number of listings with price > 500: {count_above_500}")


def main():
    """
    Main function for running tests.
    """
    # Uncomment to run different tests
    # feature_selection()
    # test_ols_model()
    # test_mystery_data()
    exploratory_analysis()


if __name__ == '__main__':
    main()