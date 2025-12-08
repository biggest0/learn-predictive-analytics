from feature_interpretation.graph import draw_heatmap, draw_hist_plots
from linear_regression.test.constants import *
from train_one import get_csv_dataframe, PATH_TO_CREDIT_CSV, PATH_TO_FULL_CLEANED_CSV

df = get_csv_dataframe(PATH_TO_FULL_CLEANED_CSV)

# list of features
selected_features = SELECTED_FEATURES
rfe_features = RFE
chi_features = CHI
ffs_features = FFS

## Generate heatmap for all 20 best features
# draw_heatmap(df, SELECTED_FEATURES)


# Generate boolean histograms for all 18 boolean features
features_boolean = SELECTED_FEATURES.copy()
features_boolean.remove('installment_commitment')
features_boolean.remove('credit_amount_(2500.0, 5000.0]')

# draw_hist_plots(df, target='class', features=features_boolean[0:3])

# Generate histogram for 2 remaining quantitative features
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import math

import seaborn as sns
import matplotlib.pyplot as plt


def boxplot_features():
    # 1. Define features suitable for boxplots
    # Boxplots work best on continuous (e.g., Age, Amount) or Ordinal (e.g., Installment Rate) data.
    # Most of your 'SELECTED_FEATURES' are binary (0/1), which don't make good boxplots.
    # strictly use meaningful continuous/ordinal features here:
    boxplot_features = [
        'installment_commitment',  # From your list (ordinal)
        'duration',  # RECOMMENDATION: Use the raw column if available
        'credit_amount',  # RECOMMENDATION: Use the raw column if available
        'age'  # RECOMMENDATION: Use the raw column if available
    ]
    target_col = 'class'

    # Filter to ensure we only plot columns that actually exist in your dataframe
    existing_features = [f for f in boxplot_features if f in df.columns]

    # 2. Setup Plot
    if existing_features:
        fig, axes = plt.subplots(len(existing_features), 1, figsize=(10, 5 * len(existing_features)))

        # Handle case where there is only one feature (axes is not a list)
        if len(existing_features) == 1:
            axes = [axes]

        for i, feature in enumerate(existing_features):
            ax = axes[i]

            # 3. Create Boxplot
            # x = Target Class, y = Continuous Feature
            sns.boxplot(
                x=target_col,
                y=feature,
                data=df,
                ax=ax,
                palette=['#d65f5f', '#9fd195'],  # Matching your Red/Green theme
                width=0.5
            )

            # Styling
            ax.set_title(f'Distribution of {feature} by Class', fontsize=12, fontweight='bold')
            ax.set_xlabel('Class (0 = Risk, 1 = No Risk)')
            ax.set_ylabel(feature)

            # Optional: Add strip plot on top to see individual data points
            # Useful if dataset is small, otherwise comment this out
            sns.stripplot(x=target_col, y=feature, data=df, ax=ax, color='black', alpha=0.3, jitter=True)

        plt.tight_layout()
        plt.show()
    else:
        print("None of the specified continuous features were found in the DataFrame.")


def proportion_plot(n, m):
    # 1. Setup your features and target
    target_col = 'class'
    selected_features = [
        'checking_status_no checking', 'checking_status_<0', 'credit_history_critical/other existing credit',
        'credit_history_no credits/all paid', 'age_(-0.001, 25.0]', 'savings_status_<100', 'duration_(36.0, 60.0]',
        'property_magnitude_no known property', 'property_magnitude_real estate', 'credit_amount_(10000.0, 20000.0]',
        'purpose_new car', 'savings_status_no known savings', 'duration_(24.0, 36.0]', 'employment_<1',
        'age_(60.0, 80.0]', 'checking_status_>=200',
        'installment_commitment', 'other_parties_guarantor', 'purpose_retraining', 'credit_amount_(2500.0, 5000.0]',
    ]
    selected_features = selected_features[n:m]
    # selected_features = [
    #     'credit_amount_(2500.0, 5000.0]', 'purpose_retraining', 'other_parties_guarantor'
    # ]
    # --- PASTE YOUR DATAFRAME HERE ---
    # Assuming your dataframe is named 'df'
    # df = ...

    # 2. Configure the Plot Grid
    num_cols = 2  # How many charts per row
    num_rows = math.ceil(len(selected_features) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 4))
    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    # 3. Iterate and Plot
    for i, feature in enumerate(selected_features):
        ax = axes[i]

        # Create a crosstab (contingency table)
        # normalize='index' converts counts to percentages (rows sum to 1)
        ct = pd.crosstab(df[feature], df[target_col], normalize='index')

        # Plot as a Stacked Bar Chart
        # Using specific colors: Red/Pink for 0, Green for 1 (matching your previous image)
        ct.plot(kind='bar', stacked=True, ax=ax, color=['#d65f5f', '#9fd195'], alpha=0.9, width=0.6)

        # Styling
        ax.set_title(f'Proportion(%) of {target_col}(1/0) for {feature}', fontsize=12)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)

        # Add a horizontal line at 50% for reference
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.tick_params(axis='x', rotation=0)


        # # Clean up legend (only show on the first plot to save space, or remove if cluttered)
        # if i == 0:
        #     ax.legend(title='Class', loc='upper right', bbox_to_anchor=(1.3, 1))
        # else:
        #     ax.get_legend().remove()

    # 4. Remove empty subplots if total features aren't a multiple of num_cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_numeric_vs_boolean(df, target='class', features=None, ncols=3, figsize=(15, 6)):
    """
    Draws side-by-side histograms showing how numeric features differ between target classes.
    """

    # Auto-select numeric features if none provided
    if features is None:
        features = df.select_dtypes(include=['int', 'float']).columns.tolist()
        if target in features:
            features.remove(target)

    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    palette = {
        1: "#7FAF7F",  # muted green
        0: "#C78787"  # muted red
    }

    for i, feature in enumerate(features):
        sns.histplot(
            data=df,
            x=feature,
            hue=target,
            multiple="dodge",  # side-by-side
            kde=False,
            palette=palette,
            edgecolor=None,
            ax=axes[i]
        )
        axes[i].set_title(f'{feature} distribution by {target}')
        axes[i].grid(True, linestyle='--', alpha=0.4)

    # Remove unused plots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_distribution_by_target(df, target='class', features=None, ncols=3, figsize=(15, 6)):
    """
    Plots boxplots showing how numeric features vary for each target class.
    """
    # Auto-select numeric features
    if features is None:
        features = df.select_dtypes(include=['int', 'float']).columns.tolist()
        if target in features:
            features.remove(target)

    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    palette = {
        1: "#7FAF7F",  # muted green
        0: "#C78787"  # muted red
    }

    for i, feature in enumerate(features):
        sns.boxplot(
            data=df,
            x=target,
            y=feature,
            palette=palette,
            ax=axes[i]
        )
        axes[i].set_title(f'{feature} by {target}')
        axes[i].grid(True, linestyle='--', alpha=0.4)

    # Remove unused plots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# plot_numeric_vs_boolean(df, features=['credit_amount_(2500.0, 5000.0]', 'installment_commitment'])


# proportion_plot()

all_3 = list(set(RFE).intersection(set(CHI)).intersection(set(FFS)))
current_RFE = all_3 + ['purpose_education']
all_2_RFE_CHI = set(FFS).intersection(set(CHI)) - set(all_3)
# print(', '.join(set(RFE).intersection(set(CHI)).intersection(set(FFS))))
# print(', '.join(all_2_RFE_CHI))
print(len(all_3))
print(len(current_RFE))
print(', '.join(all_2_RFE_CHI))
print(len(all_2_RFE_CHI))

print(set(FFS) - set(CHI))
print(', '.join(set(RFE) - set(current_RFE)))
print(len(set(RFE) - set(current_RFE)))
print(', '.join(all_3))
print(len(all_3))

print(', '.join(SELECTED_FEATURES))
print(len(SELECTED_FEATURES))

y = df['class']
counts = y.value_counts()

print(counts)
# draw_hist_plots(df, features=SELECTED_FEATURES[:3])
# proportion_plot(18, 20)

features_to_plot = [
    'checking_status_no checking', 'checking_status_<0', 'credit_history_critical/other existing credit',
    'credit_history_no credits/all paid', 'age_(-0.001, 25.0]', 'savings_status_<100', 'duration_(36.0, 60.0]',
    'property_magnitude_no known property', 'property_magnitude_real estate', 'credit_amount_(10000.0, 20000.0]',
    'purpose_new car', 'savings_status_no known savings', 'duration_(24.0, 36.0]', 'employment_<1',
    'age_(60.0, 80.0]', 'checking_status_>=200',
    'installment_commitment', 'other_parties_guarantor', 'purpose_retraining', 'credit_amount_(2500.0, 5000.0]',
]

# draw_hist_plots(df, features=features_to_plot[18:20])

df['class'].value_counts().plot(kind='bar', color=["#9fd195", "#d65f5f"])
plt.title('Target "class" (1/0) Frequency')
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()