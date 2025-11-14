import matplotlib
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np


# from util.file_handler import get_csv_dataframe
import matplotlib.pyplot as plt
from constant import TOP_FEATURES


# def test_draw_scatterplot(n):
#     df = get_csv_dataframe()
#
#     target = 'price'
#     # Get only numeric columns
#     numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#
#     # Remove target column from feature list
#     features = [col for col in numeric_cols if col != target]
#     # figure 3,4, 5, 6, 7: nothing good
#     # Dryer n = 8,
#
#     n = n * 5
#     features = features[n:n+5]
#     print(features)
#
#
#     # 1 row, 3 columns
#     fig, axes = plt.subplots(1, len(features), figsize=(15, 5))
#
#     for i, feature in enumerate(features):
#         axes[i].scatter(df[feature], df[target], alpha=0.6, edgecolor='k')
#         axes[i].set_title(f'{feature.capitalize()} vs {target.capitalize()}')
#         axes[i].set_xlabel(feature.capitalize())
#         axes[i].set_ylabel(target.capitalize())
#         axes[i].grid(True)
#
#     plt.tight_layout()
#     plt.show()


# def draw_scatterplot(features):
#     matplotlib.use('TkAgg')
#
#     df = get_csv_dataframe()
#
#     target = 'price'
#
#     fig, axes = plt.subplots(1, len(features), figsize=(15, 5))
#
#     for i, feature in enumerate(features):
#         axes[i].scatter(df[feature], df[target], alpha=0.6, edgecolor='k')
#         axes[i].set_title(f'{feature.capitalize()} vs {target.capitalize()}')
#         axes[i].set_xlabel(feature.capitalize())
#         axes[i].set_ylabel(target.capitalize())
#         axes[i].grid(True)
#
#     plt.tight_layout()
#     plt.show()


def test_draw_heatmap(df, n):
    # input a n and it shows heatmaps for n:n+10 indexes
    # df = get_csv_dataframe()

    target = 'price'
    # Get only numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Remove target column from feature list
    features = [col for col in numeric_cols if col != target]
    # accommodates 0.52, bedrooms 0.49, bathrooms 0.46, beds 0.43
    # family_kid_friendly 0.21, dryer 0.15, indoor_fireplace 0.19, tv 0.18, washer 0.15

    n = n * 10
    features = features[n:n+10]
    print(features)

    # Compute the correlation matrix
    selected_cols = features + [target]
    corr = df[selected_cols].corr()
    corr_target = corr[[target]]
    corr_target = corr_target.reindex(corr_target[target].abs().sort_values(ascending=False).index)
    # plot the heatmap


    sns.heatmap(
        corr_target,
        annot=True,          # Show numbers inside bars
        cmap="YlGnBu",
        linewidths=0.5,
        vmin=-1, vmax=1,
        cbar=True,
        fmt=".2f",
        annot_kws={"size": 10, "color": "black"}  # number color contrast
    )

    plt.title(f'Correlation with {target.capitalize()}')
    plt.xlabel('Correlation')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()


def plotPredictionVsActual(plt, title, y_test, predictions):
    plt.scatter(y_test, predictions)
    plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title('Predicted (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')

def plotResidualsVsActual(plt, title, y_test, predictions):
    residuals = y_test - predictions
    plt.scatter(y_test, residuals, label='Residuals vs Actual')
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.title('Error Residuals (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()], [0, 0], 'k--')


def drawValidationPlots(title, bins, y_test, predictions):
    # Define number of rows and columns for graph display.
    plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    plt.subplot(1, 2, 1)  # Specfy total rows, columns and image #
    plotPredictionVsActual(plt, title, y_test, predictions)

    plt.subplot(1, 2, 2)  # Specfy total rows, columns and image #
    plotResidualsVsActual(plt, title, y_test, predictions)
    plt.show()


def draw_heatmap(df, features):
    """
    Draws a correlation heatmap showing how each feature relates to the target variable 'price'.
    Negative correlations appear red; positive appear blue.
    """
    target = 'price'

    # Select relevant columns
    selected_cols = features + [target]

    # Compute correlation matrix
    corr = df[selected_cols].corr()

    # Extract correlation with the target
    corr_target = corr[[target]]

    # Sort by absolute correlation values
    corr_target = corr_target.reindex(
        corr_target[target].abs().sort_values(ascending=False).index
    )

    # Plot heatmap
    plt.figure(figsize=(6, len(features) * 0.4 + 1))
    sns.heatmap(
        corr_target,
        annot=True,
        cmap="RdYlBu",          # 🔥 Diverging color map (red = negative, blue = positive)
        linewidths=0.5,
        vmin=-1, vmax=1,
        cbar=True,
        fmt=".2f",
        annot_kws={"size": 10, "color": "black"}
    )

    plt.title(f'Correlation with {target.capitalize()}', fontsize=14)
    plt.xlabel('Correlation', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.show()




# draw validatin plots

# def plotPredictionVsActual(plt, title, y_test, predictions):
#     plt.scatter(y_test, predictions)
#     plt.legend()
#     plt.xlabel("Actual")
#     plt.ylabel("Predicted")
#     plt.title('Predicted (Y) vs. Actual (X): ' + title)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
# def plotResidualsVsActual(plt, title, y_test, predictions):
#     residuals = y_test - predictions
#     plt.scatter(y_test, residuals, label='Residuals vs Actual')
#     plt.xlabel("Actual")
#     plt.ylabel("Residual")
#     plt.title('Error Residuals (Y) vs. Actual (X): ' + title)
#     plt.plot([y_test.min(), y_test.max()], [0, 0], 'k--')
# def plotResidualHistogram(plt, title, y_test, predictions, bins):
#     residuals = y_test - predictions
#     plt.xlabel("Residual")
#     plt.ylabel("Frequency")
#     plt.hist(residuals, label='Residuals vs Actual', bins=bins)
#     plt.title('Error Residual Frequency: ' + title)
#     plt.plot()
#
# def drawValidationPlots(title, bins, y_test, predictions):
#     # Define number of rows and columns for graph display.
#     plt.subplots(nrows=1, ncols=3, figsize=(12,5))
#     plt.subplot(1, 3, 1) # Specfy total rows, columns and image #
#     plotPredictionVsActual(plt, title, y_test, predictions)
#     plt.subplot(1, 3, 2) # Specfy total rows, columns and image #
#     plotResidualsVsActual(plt, title, y_test, predictions)
#     plt.subplot(1, 3, 3) # Specfy total rows, columns and image #
#     plotResidualHistogram(plt, title, y_test, predictions, bins)
#     plt.show()


def draw_box_plot(df):

    # Rename the columns so they are more reader-friendly.
    # df = df.rename({'MomAge': 'Mom Age', 'DadAge': 'Dad Age',
    #                 'MomEduc': 'Mom Edu', 'weight': 'Weight'}, axis=1)  # new method

    # This line allows us to set the figure size supposedly in inches.
    # When rendered in the IDE the output often does not translate to inches.
    plt.subplots(nrows=1, ncols=3, figsize=(14, 7))

    plt.subplot(1, 3, 1)  # Specfies total rows, columns and image #
    # where images are drawn clockwise.
    # plt.xlabel("Mom Age")
    plt.xticks([], ())
    boxplot = df.boxplot(column=['accommodates', 'city_DC'])

    plt.subplot(1, 3, 2)  # Specfies total rows, columns and image #
    # where images are drawn clockwise.
    boxplot = df.boxplot(column=['bedrooms'])

    plt.subplot(1, 3, 3)  # Specfies total rows, columns and image #
    # where images are drawn clockwise.
    boxplot = df.boxplot(column=['bathrooms'])

    plt.show()


def _draw_box_plot(df, target='price', features=None, ncols=2, figsize=(15, 6)):
        """
        Draws boxplots showing how each feature relates to a target variable.

        Parameters:
            df (pd.DataFrame): The dataframe containing the data.
            target (str): Target column name (default = 'price').
            features (list): List of feature names to plot.
            ncols (int): Number of plots per row.
            figsize (tuple): Overall figure size.
        """

        # Auto-detect features if not provided
        if features is None:
            features = df.select_dtypes(include=['object', 'category', 'int', 'float']).columns.tolist()
            features.remove(target)

        nrows = (len(features) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()

        for i, feature in enumerate(features):
            sns.boxplot(data=df, x=feature, y=target, ax=axes[i])
            axes[i].set_title(f'{target.capitalize()} vs {feature}', fontsize=12)
            axes[i].set_xlabel(feature, fontsize=12)
            axes[i].set_ylabel(target.capitalize(), fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            # axes[i].grid(True, linestyle='--', alpha=0.5)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


def draw_hist_plots(df, target='price', features=None, ncols=3, figsize=(15, 6)):
    """
    Draws histograms for numeric features to visualize their relationship with the target.

    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        target (str): Target column name (default = 'price').
        features (list): List of feature names to plot.
        ncols (int): Number of plots per row.
        figsize (tuple): Overall figure size.
    """

    # Automatically select numeric features if none provided
    if features is None:
        features = df.select_dtypes(include=['int', 'float']).columns.tolist()
        if target in features:
            features.remove(target)

    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.histplot(
            data=df,
            x=feature,
            hue=target,
            kde=True,
            palette='viridis',
            ax=axes[i]
        )
        axes[i].set_title(f'{feature} vs {target}')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def simple_price_hist(df):
    plt.hist(df['price'], bins=30, edgecolor='black')
    plt.title('Distribution of Listing price')
    plt.xlabel('accommodates')
    plt.ylabel('price')
    plt.show()

def double_hist(df, col1='bathrooms', col2='beds'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

    # First histogram
    axes[0].hist(df[col1], bins=30, edgecolor='black')
    axes[0].set_title(f'Distribution of {col1.capitalize()}')
    axes[0].set_xlabel(col1.capitalize())
    axes[0].set_ylabel('Frequency')

    # Second histogram
    axes[1].hist(df[col2], bins=30, edgecolor='black')
    axes[1].set_title(f'Distribution of {col2.capitalize()}')
    axes[1].set_xlabel(col2.capitalize())
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

from assignment1 import train

def get_csv_dataframe(path='cleaned_AirBNB_data.csv'):
    pd.set_option('display.max_columns', None)   # show all columns
    pd.set_option('display.max_rows', None)      # show all rows (optional)
    pd.set_option('display.width', None)         # let pandas decide based on your console
    pd.set_option('display.expand_frame_repr', False)  # don't wrap to multiple lines
    return pd.read_csv(path, header = 0)

def main():
    df = get_csv_dataframe()

    TOP_FEATURES = ['accommodates', 'bedrooms', 'bathrooms', 'beds']
    MAYBE_FEATURES = ['family_kid_friendly', 'dryer', 'indoor_fireplace', 'tv', 'washer']

    features = ['cancellation_policy_super_strict_60', 'bathrooms', 'smartlock',
                'translation_missing:_en_hosting_amenity_49', 'indoor_fireplace', 'city_DC', 'doorman',
                'room_type_Private room', 'city_NYC', 'cancellation_policy_strict', 'cable_tv', 'dryer',
                'suitable_for_events', 'tv', 'family_kid_friendly', 'beds', 'bed_type_Couch', 'accommodates', 'city_SF',
                'room_type_Shared room', 'city_Chicago', 'hot_tub', 'elevator', 'bedrooms']

    # # Rename the columns so they are more reader-friendly.
    # df = df.rename({'MomAge': 'Mom Age', 'DadAge': 'Dad Age',
    #                 'MomEduc': 'Mom Edu', 'weight': 'Weight'}, axis=1)  # new method

    matplotlib.use('TkAgg')
    # df = test_draw_scatterplot(0)
    # test_draw_heatmap(df, 0)
    # _draw_box_plot(df, target='price', features=['accommodates', 'bedrooms', 'bathrooms', 'beds', 'room_type_Private room', 'family_kid_friendly', 'indoor_fireplace', 'tv', 'cable_tv', 'translation_missing:_en_hosting_amenity_49'])
    # print(df[['accommodates', 'bedrooms', 'bathrooms']].head(5000))
    # _draw_box_plot(df, target='price', features=['accommodates', 'bedrooms', 'bathrooms'])
    boxplot_features = ['accommodates', 'bedrooms']
    boxplot_features = ['bathrooms', 'beds']
    boxplot_features = ['room_type_Private room', 'family_kid_friendly', 'indoor_fireplace']
    boxplot_features = ['smartlock', 'tv', 'cable_tv']
    boxplot_features = ['dryer', 'suitable_for_events', 'city_SF']
    boxplot_features = ['room_type_Shared room', 'city_DC', 'cancellation_policy_strict']
    boxplot_features = ['city_NYC', 'hot_tub', 'cancellation_policy_super_strict_60']
    boxplot_features = ['elevator', 'city_Chicago', 'bed_type_Couch']
    boxplot_features = ['translation_missing:_en_hosting_amenity_49', 'doorman']

    # _draw_box_plot(df, target='price', features=boxplot_features)

    # HISTOGRAM STUFF
    # simple_price_hist(df)
    double_hist(df)
    # numeric_cols = ['bathrooms', 'beds', 'accommodates', 'bedrooms']
    #
    # df[numeric_cols].hist(bins=20, figsize=(10, 6), edgecolor='black')
    # plt.suptitle('Distributions of Numeric Features')
    # plt.show()

    # draw_box_plot(df)
    # draw_heatmap(df, features)
    # draw_heatmap(MAYBE_FEATURES)


if __name__ == '__main__':
    main()