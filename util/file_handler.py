import pandas as pd
from util.util import print_df_summary, create_df_subset, print_df_frequency, sort_df_by_keys, query_df_by_condition
from config import CSV_PATH
from constant import PrintFrequency, MAYBE_FEATURES, TOP_FEATURES


def get_csv_dataframe(path=CSV_PATH):
    pd.set_option('display.max_columns', None)   # show all columns
    pd.set_option('display.max_rows', None)      # show all rows (optional)
    pd.set_option('display.width', None)         # let pandas decide based on your console
    pd.set_option('display.expand_frame_repr', False)  # don't wrap to multiple lines
    return pd.read_csv(path, header = 0)


def get_pkl_dataframe():
    pd.set_option('display.max_columns', None)   # show all columns
    pd.set_option('display.max_rows', None)      # show all rows (optional)
    pd.set_option('display.width', None)         # let pandas decide based on your console
    pd.set_option('display.expand_frame_repr', False)  # don't wrap to multiple lines
    return pd.read_pickle("../imputed_data.pkl")


def get_dataframe_with_features(features):
    df = get_csv_dataframe()
    return df[features]


def get_df_imputed(pickle_path="imputed_data.pkl", df=get_csv_dataframe()):
    """
    Combines imputed numeric columns (loaded from pickle)
    with categorical columns from the given DataFrame.

    Args:
        df (pd.DataFrame): Original DataFrame with both numeric and categorical data.
        pickle_path (str): Path to the pickle file storing the imputed numeric DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame with imputed numeric and original categorical columns.
    """
    # 1. Load the imputed numeric columns from pickle
    imputed_num = pd.read_pickle(pickle_path)

    # 2. Extract categorical (non-numeric) columns from the original DataFrame
    cat_cols = df.select_dtypes(exclude='number')

    # 3. Combine numeric and categorical columns, align by index
    df_imputed = pd.concat([imputed_num, cat_cols], axis=1)

    # 4. Return the combined DataFrame
    return df_imputed


def save_local_model(df, filename):
    df.to_pickle(f"{filename}_model.pkl")


def load_local_model(pickle_path):
    return pd.read_pickle(pickle_path)


def main():
    # df = get_csv_dataframe()
    # print(df.head(10))
    #
    # print(df.describe())
    # print(df.dtypes)
    # low_df = get_dataframe_with_features(MAYBE_FEATURES)
    # print_df_summary(low_df)
    # df = create_subset_df(df, ['price', 'bathrooms', 'city'])
    # print_summary(df)
    # print_df_frequency(df, 'accommodates', PrintFrequency.KEY.value)
    # print(sort_df_by_keys(df, ['price', 'accommodates'], [False, True]).head(10))
    # print(query_df_by_condition(df, 'price >= 1950').head(10))
    df = get_pkl_dataframe()
    print(df.describe())
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    print(categorical_cols)


if __name__ == '__main__':
    main()