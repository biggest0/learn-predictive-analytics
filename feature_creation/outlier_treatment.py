import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from scipy import stats


from util.file_handler import get_csv_dataframe
import matplotlib.pyplot as plt
from constant import TOP_FEATURES
from util.file_handler import get_csv_dataframe


def find_outliers():
    dataset = get_csv_dataframe()

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    dfSub = dataset[['Avg. Area Income', 'Avg. Area House Age',
                     'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', "Area Population",
                     'Price']]
    z = np.abs(stats.zscore(dfSub))
    print(z)

    THRESHOLD = 3
    print(np.where(z > THRESHOLD))
    print(dfSub.loc[39][[0]])


def clip_outliers(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == 'price':
            continue
        mean = df[col].mean()
        std = df[col].std()

        lower_3sd = mean - 3 * std
        upper_3sd = mean + 3 * std
        lower_5sd = mean - 6 * std
        upper_5sd = mean + 6 * std

        df.loc[df[col] < lower_5sd, col] = lower_3sd
        df.loc[df[col] > upper_5sd, col] = upper_3sd
    return df
