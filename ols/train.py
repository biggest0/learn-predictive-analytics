import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from datetime import datetime
import pickle

# Your modelling code for your best model which runs from start to finish without error

PATH_TO_AIRBNB_CSV = 'AirBNB.csv'


# get dataframe from current folder
def get_csv_dataframe(csv_path=PATH_TO_AIRBNB_CSV):
    # show all columns
    pd.set_option('display.max_columns', None)
    # show all rows
    pd.set_option('display.max_rows', None)
    # let pandas decide based on your console
    pd.set_option('display.width', None)
    # don't wrap to multiple lines
    pd.set_option('display.expand_frame_repr', False)
    return pd.read_csv(csv_path, header=0)


# save
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


# imput missing data
def knn_imputer(df):
    # Create the imputer
    imputer = KNNImputer(n_neighbors=5)

    # Select only numeric columns for imputation
    numeric_df = df.select_dtypes(include=['number'])

    # Fit + transform the numeric columns
    imputed_array = imputer.fit_transform(numeric_df)

    # Put the imputed data back into a DataFrame
    imputed_df = pd.DataFrame(imputed_array, columns=numeric_df.columns)

    # Combine imputed numeric columns wit categorical columns
    categorical_df = df.select_dtypes(exclude='number')
    df = pd.concat([imputed_df, categorical_df], axis=1)

    return df


# dummy variables
def create_dummy_cols(df, col_name):
    # Create dummy variables with 0/1 row value, drop first column
    dummy_df = pd.get_dummies(df[col_name], prefix=col_name, drop_first=True, dtype=int)

    # Join back to original df and drop the original column
    df_with_dummies = pd.concat(([df, dummy_df]), axis=1)

    return df_with_dummies


def create_dummy_cols_datetime(df, col_name):
    # Convert the column to datetime format
    df[col_name] = pd.to_datetime(df[col_name], format='%d-%m-%Y', errors='coerce')

    # Create dummy variables from the datetime column
    dummy_df = pd.get_dummies(df[col_name].dt.year, prefix=col_name, drop_first=True, dtype=int)

    # Join the dummy columns back to the original df
    df_with_dummies = pd.concat(([df, dummy_df]), axis=1)

    return df_with_dummies


# binning
def create_bin_cols(df, col_name, bins, drop_original=False):
    # Bin the df column
    binned = pd.cut(df[col_name], bins=bins, include_lowest=True)

    # Create dummies for each bin (0/1)
    dummy_binned_df = pd.get_dummies(binned, prefix=col_name, dtype=int)

    # Join with original DataFrame
    df_with_bins = pd.concat(([df, dummy_binned_df]), axis=1)

    # Optionally drop original column
    if drop_original:
        df_with_bins = df_with_bins.drop(columns=[col_name])

    return df_with_bins


# clean
def clean_data():
    # Get entire df from original AirBNB csv
    df = get_csv_dataframe(PATH_TO_AIRBNB_CSV)

    # Imputs all missing numerics and combines back with categorical cols
    df = knn_imputer(df)

    # Dummy col for categorical data
    cols_to_dummy = ['city', 'bed_type', 'room_type', 'cancellation_policy', 'cleaning_fee']
    for col in cols_to_dummy:
        df = create_dummy_cols(df, col)

    # Dummy col for date time in year
    df = create_dummy_cols_datetime(df, 'host_since')

    # Bin
    df = create_bin_cols(df, 'review_scores_rating', [0, 20, 40, 60, 80, 100])

    output_path = "cleaned_AirBNB_data.csv"
    # Save cleaned df
    df.to_csv(output_path, index=False)

    return df


# train
def create_test_split(df):
    features = ['cancellation_policy_super_strict_60', 'bathrooms', 'smartlock',
                'translation_missing:_en_hosting_amenity_49', 'indoor_fireplace', 'city_DC', 'doorman',
                'room_type_Private room', 'city_NYC', 'cancellation_policy_strict', 'cable_tv', 'dryer',
                'suitable_for_events', 'tv', 'family_kid_friendly', 'beds', 'bed_type_Couch', 'accommodates', 'city_SF',
                'room_type_Shared room', 'city_Chicago', 'hot_tub', 'elevator', 'bedrooms']

    X = df[features]
    X = sm.add_constant(X)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    return X_train, y_train, X_test, y_test


# Model Summary
def summarize_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(model.summary())
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    print('Unseen Root Mean Squared Error:', rmse)
    return rmse


def k_fold(X, y, NUM_SPLITS=5):
    cv = KFold(n_splits=NUM_SPLITS, shuffle=True)
    metrics_summary = {
        "R2": [],
        "Adj_R2": [],
        "AIC": [],
        "BIC": [],
        "RMSE": []
    }

    for n, (train_indices, test_indicies) in enumerate(cv.split(X), start=1):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indicies]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indicies]

        # Fit OLS model
        model = sm.OLS(y_train, X_train).fit()

        # Predict
        preds = model.predict(X_test)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        metrics_summary["RMSE"].append(rmse)
        metrics_summary["R2"].append(model.rsquared)
        metrics_summary["Adj_R2"].append(model.rsquared_adj)
        metrics_summary["AIC"].append(model.aic)
        metrics_summary["BIC"].append(model.bic)
        print(f"Fold {n} | R²: {model.rsquared:.4f} | Adj R²: {model.rsquared_adj:.4f} | "
              f"AIC: {model.aic:.2f} | BIC: {model.bic:.2f} | RMSE: {rmse:.4f}")

    for metric, values in metrics_summary.items():
        mean_value = np.mean(values)
        sd = np.std(values) if metric == "RMSE" else None
        if sd:
            print(f"{metric}: Mean = {mean_value:.4f}, SD = {sd:.4f}")
        else:
            print(f"{metric}: Mean = {mean_value:.4f}")


def main():
    # Read CSV, impute, dummy, bin
    df = clean_data()

    # Create 85% train data, 15% unseen data
    X_train, y_train, X_test, y_test = create_test_split(df)

    # Perform kfold to get average RMSE and other metrics
    k_fold(X_train, y_train, 10)

    # Build model with 85% data, test with 15% unseen data
    final_model = sm.OLS(y_train, X_train).fit()

    # Print out unseen data summary
    rmse = summarize_model(final_model, X_test, y_test)

    # Save model and OLS summary
    save_model_and_summary(final_model, rmse)


if __name__ == '__main__':
    main()