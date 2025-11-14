import pandas as pd
from sklearn.impute import KNNImputer
import numpy  as np
from ..util.file_handler import get_csv_dataframe


def convertNAcellsToNum(colName, df, measureType):
    # Create two new column names based on original column name.
    indicatorColName = 'm_'   + colName # Tracks whether imputed.
    imputedColName   = 'imp_' + colName # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputedValue = 0
    if(measureType=="median"):
        imputedValue = df[colName].median()
    elif(measureType=="mode"):
        imputedValue = float(df[colName].mode())
    else:
        imputedValue = df[colName].mean()

    # Populate new columns with data.
    imputedColumn  = []
    indictorColumn = []
    for i in range(len(df)):
        isImputed = False

        # mi_OriginalName column stores imputed & original data.
        if(np.isnan(df.loc[i][colName])):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.loc[i][colName])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if(isImputed):
            indictorColumn.append(1)
        else:
            indictorColumn.append(0)

    # Append new columns to dataframe but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName]   = imputedColumn
    return df

# imputes all missing numeric columns
def KNN_imputer(df):
    # Step 1: Create the imputer
    imputer = KNNImputer(n_neighbors=5)

    # Step 2: Select only numeric columns for imputation
    numeric_df = df.select_dtypes(include=['number'])

    # Step 3: Fit + transform the numeric columns
    imputed_array = imputer.fit_transform(numeric_df)

    # Step 4: Put the imputed data back into a DataFrame
    imputed_df = pd.DataFrame(imputed_array, columns=numeric_df.columns)

    imputed_df.to_pickle("imputed_data.pkl")

    print(imputed_df.describe())

df = get_csv_dataframe()
KNN_imputer(df)
