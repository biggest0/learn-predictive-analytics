import pandas as pd

def create_dummy(df, column):
    encoded_df = pd.get_dummies(df, columns=[column])
    return encoded_df
