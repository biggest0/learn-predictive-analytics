import pandas as pd
from util import print_summary, create_subset_df, print_frequency, sort_df_by_keys, query_df_by_condition
from config import CSV_PATH
from constant import PrintFrequency

def get_csv_dataframe():
    pd.set_option('display.max_columns', None)   # show all columns
    pd.set_option('display.max_rows', None)      # show all rows (optional)
    pd.set_option('display.width', None)         # let pandas decide based on your console
    pd.set_option('display.expand_frame_repr', False)  # don't wrap to multiple lines
    return pd.read_csv(CSV_PATH, header = 0)


def main():
    df = get_csv_dataframe()
    # print(df.head(10))
    #
    # print(df.describe())
    # print(df.dtypes)

    # print_summary(df)
    # df = create_subset_df(df, ['price', 'bathrooms', 'city'])
    print_summary(df)
    # print_frequency(df, 'price', PrintFrequency.KEY.value)
    print(sort_df_by_keys(df, ['price', 'accommodates'], [False, True]).head(10))
    print(query_df_by_condition(df, 'price >= 1950').head(10))


if __name__ == '__main__':
    main()