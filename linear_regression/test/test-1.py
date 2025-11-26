import pandas as pd
from sklearn.impute import KNNImputer

PATH_TO_AIRBNB_CSV = 'Credit_Train.csv'


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


df = get_csv_dataframe()
df = knn_imputer(df)
df = create_dummy_cols(df, 'checking_status')
print(df.head(10))
print(df.describe())
print(df.info())

numeric_cols_to_impute = ['credit_amount'] # 713/750
categorical_cols_to_impute = ['checking_status'] # 710/750