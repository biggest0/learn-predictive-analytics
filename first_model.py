import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
import numpy as np

from file_handler import get_dataframe_with_features, get_csv_dataframe
from constant import TOP_FEATURES, MAYBE_FEATURES
from impute import convertNAcellsToNum


def ols():
    df = get_csv_dataframe()
    df = convertNAcellsToNum('bathrooms', df, "mean")
    df = convertNAcellsToNum('bedrooms', df, "mean")
    df = convertNAcellsToNum('beds', df, "mean")

    features = ['accommodates', 'imp_bedrooms', 'imp_bathrooms', 'imp_beds']
    test_features = ['dryer', 'family_kid_friendly', 'indoor_fireplace', 'tv']
    # dryer, family_kid_friendly
    features = features + test_features
    X = df[features]
    # X = df[['accommodates', 'imp_bedrooms', 'imp_bathrooms', 'imp_beds', 'family_kid_friendly']]
    print(X.head(10))
    # print(df.head(10))

    X = sm.add_constant(X)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model
    print(model.summary())
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    str_feature = ', '.join(features)

    with open(f"data/ols_summary/ols_{str_feature}.txt", "w") as f:
        f.write(model.summary().as_text())
        f.write(f'\nSelected features: {str_feature}')
        f.write(f'\nRoot Mean Squared Error:, {np.sqrt(metrics.mean_squared_error(y_test, predictions))}')

def main():
    ols()

if __name__ == '__main__':
    main()