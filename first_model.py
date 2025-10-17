import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LogisticRegression

from file_handler import get_dataframe_with_features, get_csv_dataframe
from constant import TOP_FEATURES, MAYBE_FEATURES
from impute import convertNAcellsToNum


def ols():
    df = get_csv_dataframe()
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

    no_scaling_linear_prediction(X_train, y_train, X_test, y_test)

    # # OLS print output
    # print(model.summary())
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    #
    # # Save OLS summary to .txt
    # str_feature = ', '.join(features)
    # with open(f"data/ols_summary/ols_{str_feature}.txt", "w") as f:
    #     f.write(model.summary().as_text())
    #     f.write(f'\nSelected features: {str_feature}')
    #     f.write(f'\nRoot Mean Squared Error:, {np.sqrt(metrics.mean_squared_error(y_test, predictions))}')


def no_scaling_logistic_prediction(X_train, y_train, X_test, y_test):
    # Perform logistic regression.
    model = LogisticRegression(fit_intercept=True, solver='liblinear')

    # Fit the model.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Show confusion matrix and accuracy scores.
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix")
    print(cm)


def no_scaling_linear_prediction(X_train, y_train, X_test, y_test):
    # Perform linear regression
    model = LinearRegression()

    # Fit the model.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate regression performance
    print("\nR² (Coefficient of Determination):", r2_score(y_test, y_pred))
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))

    # Optional: compare actual vs predicted values
    comparison = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    print("\nActual vs Predicted:")
    print(comparison.head())

def main():
    ols()

if __name__ == '__main__':
    main()