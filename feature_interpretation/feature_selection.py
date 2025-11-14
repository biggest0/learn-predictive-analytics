import pandas as pd
from sklearn.feature_selection import f_regression, RFE
from util.file_handler import get_pkl_dataframe
from sklearn.linear_model import LinearRegression

def forward_feature_selection(df):
    X = df.copy()  # Create separate copy to prevent unwanted tampering of data.
    del X['price']  # Delete target variable.

    # Target variable
    y = df['price']
    #  f_regression returns F statistic for each feature.
    ffs = f_regression(X, y)

    featuresDf = pd.DataFrame()
    for i in range(0, len(X.columns)):
        featuresDf = featuresDf._append({"feature": X.columns[i],
                                         "ffs": ffs[0][i]}, ignore_index=True)
    featuresDf = featuresDf.sort_values(by=['ffs'], ascending=False)

    # Get top 20 features
    top_20_features = featuresDf.head(20)
    print(top_20_features)
    top_features_list = top_20_features['feature'].tolist()
    print(top_features_list)


def recursive_feature_elimination(df):
    # Seperate the target and independent variable
    X = df.copy()  # Create separate copy to prevent unwanted tampering of data.
    del X['price']  # Delete target variable.

    # Target variable
    y = df['price']

    # Create the object of the model
    model = LinearRegression()

    # Specify the number of  features to select
    rfe = RFE(model, n_features_to_select=20)

    # fit the model
    rfe = rfe.fit(X, y)
    # Please uncomment the following lines to see the result
    print('\n\nFEATUERS SELECTED\n\n')
    print(rfe.support_)

    top_features_list = []
    columns = list(X.keys())
    for i in range(0, len(columns)):
        if (rfe.support_[i]):
            print(columns[i])
            top_features_list.append(columns[i])

    print(top_features_list)

# df = get_pkl_dataframe()
# forward_feature_selection(df)
# recursive_feature_elimination(df)

