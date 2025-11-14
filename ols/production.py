import pandas as pd
from sklearn.impute import KNNImputer


PATH_TO_AIRBNB_CSV = 'AirBNB_mystery.csv'


# Get dataframe
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


# Save prediction
def save_prediction_csv(predictions, csv_path='AirBNB_predictions.csv'):
    # Convert list of predictions to a DataFrame with column 'price'
    df = pd.DataFrame({'price': predictions})

    # Save to CSV (no index column)
    df.to_csv(csv_path, index=False)
    print('Successfully saved to AirBNB_predictions.csv')


# Imput missing data
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


# Dummy variables
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


# Binning
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


# Fill in missing dummy columns
def add_missing_cols(dfTrainTestPrep, dfProduction):
    trainTestColumns = dfTrainTestPrep
    productionColumns = list(dfProduction.keys())
    count = 0
    not_in = 0
    for col in trainTestColumns:
        if col in productionColumns:
            count += 1
        else:
            not_in += 1
    for i in range(0, len(trainTestColumns)):
        columnFound = False
        for j in range(0, len(productionColumns)):
            if (trainTestColumns[i] == productionColumns[j]):
                columnFound = True
                break
        # Add column and store zeros in every cell if
        # not found.
        if (not columnFound):
            colName = trainTestColumns[i]
            dfProduction[colName] = 0
    return dfProduction


# Clean
def clean_data():
    # Change path to csv if needed
    mystery_df = get_csv_dataframe(PATH_TO_AIRBNB_CSV)

    features = ['cancellation_policy_super_strict_60', 'bathrooms', 'smartlock',
                'translation_missing:_en_hosting_amenity_49', 'indoor_fireplace', 'city_DC', 'doorman',
                'room_type_Private room', 'city_NYC', 'cancellation_policy_strict', 'cable_tv', 'dryer',
                'suitable_for_events', 'tv', 'family_kid_friendly', 'beds', 'bed_type_Couch', 'accommodates', 'city_SF',
                'room_type_Shared room', 'city_Chicago', 'hot_tub', 'elevator', 'bedrooms']

    # Imputs all missing numerics and combines back with categorical cols
    mystery_df = knn_imputer(mystery_df)

    # Dummy col for categorical data
    cols_to_dummy = ['city', 'bed_type', 'room_type', 'cancellation_policy', 'cleaning_fee']
    for col in cols_to_dummy:
        mystery_df = create_dummy_cols(mystery_df, col)

    # Dummy col for date time in year
    mystery_df = create_dummy_cols_datetime(mystery_df, 'host_since')

    # Bin
    mystery_df = create_bin_cols(mystery_df, 'review_scores_rating', [0, 20, 40, 60, 80, 100])

    # Add missing columns if any
    mystery_df = add_missing_cols(features, mystery_df)

    X_mystery = mystery_df[features]
    return X_mystery


# Predict
def predict_price(cancellation_policy_super_strict_60, bathrooms, smartlock, en_hosting_amenity_49, indoor_fireplace,
                  city_DC, doorman, room_type_Private_room, city_NYC, cancellation_policy_strict, cable_tv, dryer,
                  suitable_for_events, tv, family_kid_friendly, beds, bed_type_Couch, accommodates, city_SF,
                  room_type_Shared_room, city_Chicago, hot_tub, elevator, bedrooms):
    const = 1.0516
    x1 = 680.2175 * cancellation_policy_super_strict_60
    x2 = 65.0470 * bathrooms
    x3 = -43.4803 * smartlock
    x4 = -9.5080 * en_hosting_amenity_49
    x5 = 25.5112 * indoor_fireplace
    x6 = 55.1362 * city_DC
    x7 = 27.8321 * doorman
    x8 = -71.7375 * room_type_Private_room
    x9 = 15.8492 * city_NYC
    x10 = -4.8809 * cancellation_policy_strict
    x11 = 7.0587 * cable_tv
    x12 = 0.9378 * dryer
    x13 = 33.6227 * suitable_for_events
    x14 = 6.6385 * tv
    x15 = -13.2581 * family_kid_friendly
    x16 = -11.2960 * beds
    x17 = 33.2805 * bed_type_Couch
    x18 = 17.6033 * accommodates
    x19 = 75.2165 * city_SF
    x20 = -100.4122 * room_type_Shared_room
    x21 = -28.3487 * city_Chicago
    x22 = 13.1411 * hot_tub
    x23 = 33.7346 * elevator
    x24 = 39.6810 * bedrooms

    price = sum(
        [const, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22,
         x23, x24])

    print(f"Price Predicted: {price}")
    return price


def predictions_price(X_mystery):
    predictions = []

    # Iterate through each row in the DataFrame
    for _, row in X_mystery.iterrows():
        # Pass each column value to your predict_price() function
        price = predict_price(
            cancellation_policy_super_strict_60=row['cancellation_policy_super_strict_60'],
            bathrooms=row['bathrooms'],
            smartlock=row['smartlock'],
            en_hosting_amenity_49=row['translation_missing:_en_hosting_amenity_49'],
            indoor_fireplace=row['indoor_fireplace'],
            city_DC=row['city_DC'],
            doorman=row['doorman'],
            room_type_Private_room=row['room_type_Private room'],
            city_NYC=row['city_NYC'],
            cancellation_policy_strict=row['cancellation_policy_strict'],
            cable_tv=row['cable_tv'],
            dryer=row['dryer'],
            suitable_for_events=row['suitable_for_events'],
            tv=row['tv'],
            family_kid_friendly=row['family_kid_friendly'],
            beds=row['beds'],
            bed_type_Couch=row['bed_type_Couch'],
            accommodates=row['accommodates'],
            city_SF=row['city_SF'],
            room_type_Shared_room=row['room_type_Shared room'],
            city_Chicago=row['city_Chicago'],
            hot_tub=row['hot_tub'],
            elevator=row['elevator'],
            bedrooms=row['bedrooms']
        )

        predictions.append(price)
    return predictions


def main():
    X_mystery = clean_data()
    predictions = predictions_price(X_mystery)
    save_prediction_csv(predictions)



if __name__ == '__main__':
    main()
