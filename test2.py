from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import statsmodels.api as sm
import numpy as np
from datetime import datetime
import pickle

from util.file_handler import get_pkl_dataframe
from constant import TOP_20_FEATURE_RECURSIVE

from feature_creation.dummy import create_dummy_cols, create_dummy_cols_datetime
from feature_creation.bin import create_bin_cols
from feature_creation.outlier_treatment import clip_outliers
from util.file_handler import get_csv_dataframe, get_df_imputed, save_local_model, load_local_model
from feature_interpretation.interpret_features import get_top_by_col
from feature_interpretation.feature_selection import forward_feature_selection, recursive_feature_elimination
from feature_interpretation.k_fold import run_kfold
from constant import *

from assignment1.train import get_csv_dataframe, knn_imputer, create_dummy_cols, create_dummy_cols_datetime, create_bin_cols

def save_model_and_summary(model, rmse):
    # Get all feature names used in the model
    all_features = model.params.index.tolist()

    # Remove 'const' (intercept) if you want only actual features
    features = [f for f in all_features if f != 'const']

    str_feature = ', '.join(features)
    now = datetime.now().time()

    with open(f"model_{now}_{rmse:.2f}.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(f"ols_{now}_{rmse:.2f}.txt", "w") as f:
        f.write(model.summary().as_text())
        f.write(f'\nSelected features: {str_feature}')
        f.write(f'\nRoot Mean Squared Error:, {rmse}')


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

def clean_data():
    df = get_csv_dataframe('cleaned_AirBNB_data.csv')

    # Imputs all missing numerics and combines back with categorical cols
    df = knn_imputer(df)

    # Dummy col for categorical data
    cols_to_dummy = ['city', 'bed_type', 'room_type', 'cancellation_policy', 'cleaning_fee', 'neighbourhood']
    for col in cols_to_dummy:
        df = create_dummy_cols(df, col)

    # Dummy col for date time in year
    df = create_dummy_cols_datetime(df, 'host_since')

    # Bin
    df = create_bin_cols(df, 'review_scores_rating', [0, 20, 40, 60, 80, 100])

    output_path = "cleaned_AirBNB_data.csv"
    df.to_csv(output_path, index=False)

    return df

# df = clean_data()
df = get_csv_dataframe('cleaned_AirBNB_data.csv')
# df_numeric = df.select_dtypes(include='number')
# forward_feature_selection(df_numeric)
# recursive_feature_elimination(df_numeric)


features_one = ['cancellation_policy_super_strict_60', 'neighbourhood_Bel Air/Beverly Crest', 'neighbourhood_Bellevue', 'neighbourhood_Chevy Chase, MD', 'neighbourhood_Cow Hollow', 'neighbourhood_Fort Totten', 'neighbourhood_Foxhall', 'neighbourhood_Gallaudet', 'neighbourhood_Gateway', 'neighbourhood_Judiciary Square', 'neighbourhood_Laurel Canyon', 'neighbourhood_Malibu', 'neighbourhood_Pacific Palisades', 'neighbourhood_Presidio Heights', 'neighbourhood_Sea Cliff', 'neighbourhood_Skyland', 'neighbourhood_Tribeca', 'neighbourhood_Van Nest', 'neighbourhood_West Athens', 'neighbourhood_Wilmington']
features_two = ['accommodates', 'bedrooms', 'bathrooms', 'beds', 'room_type_Private room', 'family_kid_friendly', 'indoor_fireplace', 'tv', 'cable_tv', 'translation_missing:_en_hosting_amenity_49', 'dryer', 'washer', 'suitable_for_events', 'city_SF', 'neighbourhood_Malibu', 'room_type_Shared room', 'city_DC', 'cancellation_policy_strict', 'city_NYC', 'hot_tub']

features_forward = set(features_two)
features_recursive = set(features_one)

# OR if you want all unique features (union)
features = list(features_forward.union(features_recursive))

features_to_remove = ['neighbourhood_Gateway', 'dryer', 'neighbourhood_Foxhall', 'neighbourhood_Fort Totten', 'neighbourhood_West Athens']
for feature in features_to_remove:
    features.remove(feature)

X = df[features]

X = sm.add_constant(X)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)  # make the predictions by the model
print(model.summary())
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', rmse)

run_kfold(X_train, y_train, 10)