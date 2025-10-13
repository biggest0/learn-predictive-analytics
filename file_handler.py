import pandas as pd

from config import CSV_PATH

df = pd.read_csv(CSV_PATH, header = 0)
print(df.head(10))