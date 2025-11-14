import pandas as pd


def print_df_summary(df):
    pd.set_option('display.max_columns', None)  # show all columns
    pd.set_option('display.max_rows', None)  # show all rows (optional)
    pd.set_option('display.width', None)  # let pandas decide based on your console
    pd.set_option('display.expand_frame_repr', False)  # don't wrap to multiple lines


    print('Statistical Summary of numbers\n', df.describe().round(2), '\n --------------------')
    print('Statistical Summary of string/ dateTime\n', df.describe(include=['object']), '\n --------------------')
    print('Column data type\n', df.dtypes, '\n--------------------')


def print_df_frequency(df, key, sort):
    if sort == 'descending':
        print(f'Frequency: Highest first ({key})\n', df[key].value_counts(), '\n --------------------')
    elif sort == 'ascending':
        print(f'Frequency: Lowest first ({key})\n', df[key].value_counts(ascending=True), '\n --------------------')
    else:
        print(f'Frequency: Sorted by ({key})\n', df[key].value_counts().sort_index(),'\n --------------------')


def sort_df_by_keys(df, keys, order):
    return df.sort_values(by=keys, ascending=order).reset_index(drop=True)


def query_df_by_condition(df, condition):
    # condition is a string: 'DadAge >= 40 and MomAge >= 40'
    return df.query(condition)


def create_df_subset(df, keys):
    return df[keys]
