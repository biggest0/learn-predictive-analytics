"""
interpret df functions
"""


def get_top_by_col(df, col, count):
    top_20 = df.sort_values(by=col, ascending=False).head(count)
    print(top_20)