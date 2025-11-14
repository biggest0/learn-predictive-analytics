from sklearn.model_selection import train_test_split
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

def feature_selection():
    # df = get_csv_dataframe()
    df = get_df_imputed()

    # prints all categorical cols
    # categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    # print(categorical_cols)

    # get_top_by_col(df, 'price', 2)

    for dummy in COLUMNS_TO_DUMMY:
        df = create_dummy_cols(df, dummy)

    df = create_dummy_cols_datetime(df, 'host_since')

    df = create_bin_cols(df, 'review_scores_rating', [0, 20, 40, 60, 80, 100])
    save_local_model(df, 'test')

    df_numeric = df.select_dtypes(include='number')

    # forward_feature_selection(df_numeric)
    recursive_feature_elimination(df_numeric)



    print(df.head(10))

def test_ols_model():
    # features = TWO_TOP_20_FEATURE_FORWARD
    # features = TWO_TOP_20_FEATURE_RECURSIVE

    # Assuming these are lists
    features_forward = set(TWO_TOP_20_FEATURE_FORWARD)
    features_recursive = set(TWO_TOP_20_FEATURE_RECURSIVE)

    # Find common features (intersection)
    features = list(features_forward.intersection(features_recursive))

    # OR if you want all unique features (union)
    features = list(features_forward.union(features_recursive))

    # features_to_remove = ['air_purifier', 'beachfront', 'paid_parking_off_premises', 'path_to_entrance_lit_at_night', 'roll_in_shower_with_chair', 'free_parking_on_street']
    features_to_remove = ['roll_in_shower_with_chair', 'washer', 'air_purifier']
    features_to_remove += ['beachfront', 'path_to_entrance_lit_at_night']
    features_to_remove += ['free_parking_on_street', 'paid_parking_off_premises']
    # features_to_remove = []
    for feature in features_to_remove:
        features.remove(feature)
    df = load_local_model('test_model.pkl')
    df = clip_outliers(df)
    print(df.head(10))
    X = df[features]

    X = sm.add_constant(X)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model
    print(model.summary())
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error:', rmse)

    run_kfold(X, y, 10)

    # # Apply log transform to target variable
    # y = np.log1p(df['price'])  # log(1 + price)
    #
    # # Train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #
    # # Fit OLS model
    # model = sm.OLS(y_train, X_train).fit()
    # predictions_log = model.predict(X_test)
    #
    # # Convert predictions back to original scale
    # predictions = np.expm1(predictions_log)
    # y_test_original = np.expm1(y_test)
    #
    # # Print model summary
    # print(model.summary())
    #
    # # Calculate RMSE in original scale
    # rmse = np.sqrt(metrics.mean_squared_error(y_test_original, predictions))
    # print('Root Mean Squared Error (Original Scale):', rmse)

    str_feature = ', '.join(features)
    now = datetime.now().time()

    save_dir = "data/pickle_model"

    with open(f"{save_dir}/model_{now}_{rmse:.2f}.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(f"data/ols_summary/ols_{now}_{rmse:.2f}.txt", "w") as f:
        f.write(model.summary().as_text())
        f.write(f'\nSelected features: {str_feature}')
        f.write(f'\nRoot Mean Squared Error:, {np.sqrt(metrics.mean_squared_error(y_test, predictions))}')


def make_missing_cols(dfTrainTestPrep, dfProduction):
    trainTestColumns = dfTrainTestPrep
    print(trainTestColumns)
    productionColumns = list(dfProduction.keys())
    print(productionColumns)
    for i in range(0, len(trainTestColumns)):
        columnFound = False
        for j in range(0, len(productionColumns)):
            if(trainTestColumns[i]==productionColumns[j]):
                columnFound = True
                break
        # Add column and store zeros in every cell if
        # not found.
        if(not columnFound):
            colName = trainTestColumns[i]
            dfProduction[colName] = 0
    return dfProduction


# def calculate_price():



def test_mystery_data():
    model = load_local_model('data/pickle_model/model_23:40:24.621383_126.16.pkl')

    mystery = get_csv_dataframe('./data/AirBNB_mystery.csv')

    for dummy in COLUMNS_TO_DUMMY:
        mystery = create_dummy_cols(mystery, dummy)

    mystery = create_dummy_cols_datetime(mystery, 'host_since')

    mystery = create_bin_cols(mystery, 'review_scores_rating', [0, 20, 40, 60, 80, 100])
    make_missing_cols(model.model.exog_names, mystery)
    print(mystery.head(10))

    feature_cols = model.model.exog_names
    print(feature_cols)
    X_mystery = mystery[feature_cols].copy()

    X_mystery = sm.add_constant(X_mystery)

    predictions = model.predict(X_mystery)

    print("Predictions for mystery data:")
    print(predictions.head(10))

    # features = ['smartlock', 'beds', 'translation_missing:_en_hosting_amenity_49', 'cancellation_policy_strict', 'suitable_for_events', 'room_type_Shared room', 'tv', 'city_SF', 'bed_type_Couch', 'cancellation_policy_super_strict_60', 'city_Chicago', 'cable_tv', 'hot_tub', 'indoor_fireplace', 'bedrooms', 'city_NYC', 'city_DC', 'accommodates', 'family_kid_friendly', 'elevator', 'room_type_Private room', 'bathrooms', 'doorman']
    # X = df[features]

# test_ols_model()
# feature_selection()
# test_mystery_data()

df = get_csv_dataframe()
prices = df['price']
count_below_750 = (prices > 500).sum()
print(count_below_750)