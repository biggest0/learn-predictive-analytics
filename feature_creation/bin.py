import pandas as pd
import numpy as np


def create_bin_cols(df, col_name, bins, drop_original=False):
    """
    Create separate 0/1 columns for each bin range of a numeric column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the column.
        col_name (str): The numeric column to bin.
        bins (list): List of bin edges, e.g. [0, 20, 40, 60, 80, 100].
        drop_original (bool): Whether to drop the original column.

    Returns:
        pd.DataFrame: DataFrame with new binary columns for each bin.
    """
    # Bin the data using pd.cut
    binned = pd.cut(df[col_name], bins=bins, include_lowest=True)

    # Get dummies for each bin (0/1 columns)
    dummies = pd.get_dummies(binned, prefix=col_name)

    # Convert to 0/1 explicitly
    dummies = dummies.astype(int)

    # Join with original DataFrame
    df_with_bins = df.join(dummies)

    # Optionally drop original column
    if drop_original:
        df_with_bins = df_with_bins.drop(columns=[col_name])

    return df_with_bins


def _create_bin_cols(df, col_name, bins, labels=None, drop_original=False):
    """
    Dynamically create binned columns for a numeric column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the column.
        col_name (str): The name of the numeric column to bin.
        bins (list or int): If list → bin edges; if int → number of bins.
        labels (list, optional): Labels for bins. If None, numeric labels are used.
        drop_original (bool): Whether to drop the original column after binning.

    Returns:
        pd.DataFrame: DataFrame with the new binned column added.
    """
    # Create binned column name
    binned_col_name = f"{col_name}_binned"

    # Use pd.cut for binning
    df[binned_col_name] = pd.cut(df[col_name], bins=bins, labels=labels)

    # Optionally drop original column
    if drop_original:
        df = df.drop(columns=[col_name])

    return df
