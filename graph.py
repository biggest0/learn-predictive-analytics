import matplotlib
import numpy as np
import seaborn as sns

from file_handler import get_csv_dataframe
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from constant import TOP_FEATURES, MAYBE_FEATURES


def n_draw_scatterplot(n):
    df = get_csv_dataframe()

    target = 'price'
    # Get only numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Remove target column from feature list
    features = [col for col in numeric_cols if col != target]
    # figure 3,4, 5, 6, 7: nothing good
    # Dryer n = 8,

    n = n * 5
    features = features[n:n+5]
    print(features)


    # 1 row, 3 columns
    fig, axes = plt.subplots(1, len(features), figsize=(15, 5))

    for i, feature in enumerate(features):
        axes[i].scatter(df[feature], df[target], alpha=0.6, edgecolor='k')
        axes[i].set_title(f'{feature.capitalize()} vs {target.capitalize()}')
        axes[i].set_xlabel(feature.capitalize())
        axes[i].set_ylabel(target.capitalize())
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


def draw_scatterplot(features):
    matplotlib.use('TkAgg')

    df = get_csv_dataframe()

    target = 'price'

    fig, axes = plt.subplots(1, len(features), figsize=(15, 5))

    for i, feature in enumerate(features):
        axes[i].scatter(df[feature], df[target], alpha=0.6, edgecolor='k')
        axes[i].set_title(f'{feature.capitalize()} vs {target.capitalize()}')
        axes[i].set_xlabel(feature.capitalize())
        axes[i].set_ylabel(target.capitalize())
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


def n_draw_heatmap(n):
    # input a n and it shows heatmaps for n:n+10 indexes
    df = get_csv_dataframe()

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


def draw_heatmap(features):
    df = get_csv_dataframe()

    target = 'price'

    # Compute the correlation matrix
    selected_cols = features + [target]

    # creates corr matrix of all the selected features
    corr = df[selected_cols].corr()
    # gets the col of corr values for the specific target
    # it is a single col now of Price correlation with the other features
    corr_target = corr[[target]]

    # sorts col based on absolute ratio of corr
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


def main():
    matplotlib.use('TkAgg')
    # df = test_draw_scatterplot(0)
    # test_draw_heatmap(0)
    draw_heatmap(TOP_FEATURES)
    # draw_heatmap(MAYBE_FEATURES)


if __name__ == '__main__':
    main()