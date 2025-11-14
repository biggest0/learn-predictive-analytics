import pandas as pd


def create_dummy_cols(df, col_name):
    # Create dummy variables, drop_first=True to avoid multicollinearity
    dummies_dropped = pd.get_dummies(df[col_name], prefix=col_name, drop_first=True)

    # Ensure 0/1 integers instead of True/False
    dummies_dropped = dummies_dropped.astype(int)

    # Join back to original df and drop the original column
    df_with_dummies = df.join(dummies_dropped)
    df_with_dummies = df_with_dummies.drop(col_name, axis=1)

    # print(df_with_dummies.head(10))
    return df_with_dummies


def create_dummy_cols_datetime(df, col_name):
    # Convert the column to datetime format
    df[col_name] = pd.to_datetime(df[col_name], format='%d-%m-%Y', errors='coerce')

    # Create dummy variables from the datetime column (year/month/day etc.)
    dummies = pd.get_dummies(df[col_name].dt.year, prefix=col_name, drop_first=True)

    # Convert True/False to 0/1
    dummies = dummies.astype(int)

    # Join the dummy columns back to the original df
    df_with_dummies = df.join(dummies)

    # Optionally drop the original datetime column
    df_with_dummies = df_with_dummies.drop(col_name, axis=1)

    return df_with_dummies
