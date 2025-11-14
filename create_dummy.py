import pandas as pd

def create_dummy(df, column):
    encoded_df = pd.get_dummies(df, columns=[column])
    encoded_df = encoded_df.astype(int)
    return encoded_df
