import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from linear_regression.test.constants import SELECTED_FEATURES
from linear_regression.test.train_one import get_csv_dataframe

df = get_csv_dataframe('full_cleaned_data.csv')


def one_big_PCA_decision_boundary():
    # 1. SETUP & MOCK DATA GENERATION
    # ---------------------------------------------------------
    # Since I don't have your actual dataset, I will generate synthetic data
    # that mimics the structure of your 20 selected features.

    SELECTED_FEATURES = [
        'checking_status_no checking', 'checking_status_<0',
        'credit_history_critical/other existing credit',
        'age_(-0.001, 25.0]', 'duration_(36.0, 60.0]', 'credit_amount_(10000.0, 20000.0]',
        'checking_status_>=200', 'purpose_new car', 'other_parties_guarantor', 'duration_(24.0, 36.0]',
        'savings_status_<100', 'purpose_retraining', 'age_(60.0, 80.0]', 'property_magnitude_real estate',
        'savings_status_no known savings', 'installment_commitment', 'credit_amount_(2500.0, 5000.0]',
        'employment_<1', 'property_magnitude_no known property', 'credit_history_no credits/all paid'
    ]

    target_col = 'class'

    # Generate synthetic data (1000 samples, 20 features)
    # We use make_classification to ensure there is a pattern to find
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15, # High information content to match complex credit data
        n_redundant=5,
        random_state=42
    )

    # Convert to DataFrame for realism
    df = pd.DataFrame(X, columns=SELECTED_FEATURES)
    df[target_col] = y

    print("Training Logistic Regression on 20 features...")

    # 2. TRAIN THE MODEL
    # ---------------------------------------------------------
    # Standardize features (important for Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[SELECTED_FEATURES])

    # Fit the actual high-dimensional model
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, df[target_col])

    # 3. CALCULATE METRICS
    # ---------------------------------------------------------
    y_pred = model.predict(X_scaled)

    print("\nModel Performance Metrics:")
    print("-" * 30)
    print(f"Accuracy:  {accuracy_score(df[target_col], y_pred):.4f}")
    print(f"Precision: {precision_score(df[target_col], y_pred):.4f}")
    print(f"Recall:    {recall_score(df[target_col], y_pred):.4f}")
    print(f"F1 Score:  {f1_score(df[target_col], y_pred):.4f}")
    print("-" * 30)
    print("\nFull Classification Report:")
    print(classification_report(df[target_col], y_pred))

    # 4. DIMENSIONALITY REDUCTION (PCA) FOR VISUALIZATION
    # ---------------------------------------------------------
    # We cannot plot 20 dimensions. We reduce to 2 dimensions using PCA.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 5. CREATE THE DECISION BOUNDARY
    # ---------------------------------------------------------
    # We create a meshgrid covering the 2D PCA space
    h = .02  # Step size in the mesh
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # CRITICAL STEP:
    # To visualize the *original* model's boundary, we must project the
    # 2D meshgrid points back into the original 20D space.
    mesh_points_2d = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_20d = pca.inverse_transform(mesh_points_2d)

    # Predict using the original 20-feature model
    Z = model.predict(mesh_points_20d)
    Z = Z.reshape(xx.shape)

    # 6. PLOTTING
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))

    # Plot the contour (decision boundary)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # Plot the actual data points
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df[target_col],
                          edgecolor='k', cmap=plt.cm.coolwarm, s=50, alpha=0.7)

    # Add details
    plt.title('Logistic Regression Decision Boundary\n(Projected from 20D to 2D via PCA)', fontsize=14)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Plot generated successfully.")
    print(f"Explained Variance by 2 components: {np.sum(pca.explained_variance_ratio_):.2%}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def twenty_sigmoid_curves():
    # 1. SETUP & MOCK DATA GENERATION
    # ---------------------------------------------------------
    # Since I don't have your actual dataset, I will generate synthetic data
    # that mimics the structure of your 20 selected features.

    SELECTED_FEATURES = [
        'checking_status_no checking', 'checking_status_<0',
        'credit_history_critical/other existing credit',
        'age_(-0.001, 25.0]', 'duration_(36.0, 60.0]', 'credit_amount_(10000.0, 20000.0]',
        'checking_status_>=200', 'purpose_new car', 'other_parties_guarantor', 'duration_(24.0, 36.0]',
        'savings_status_<100', 'purpose_retraining', 'age_(60.0, 80.0]', 'property_magnitude_real estate',
        'savings_status_no known savings', 'installment_commitment', 'credit_amount_(2500.0, 5000.0]',
        'employment_<1', 'property_magnitude_no known property', 'credit_history_no credits/all paid'
    ]

    target_col = 'class'

    # Generate synthetic data (1000 samples, 20 features)
    # We use make_classification to ensure there is a pattern to find
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,  # High information content to match complex credit data
        n_redundant=5,
        random_state=42
    )

    # Convert to DataFrame for realism
    df = pd.DataFrame(X, columns=SELECTED_FEATURES)
    df[target_col] = y

    print("Training Logistic Regression on 20 features (Full Model)...")

    # 2. TRAIN THE FULL MODEL (For Metrics Only)
    # ---------------------------------------------------------
    # We still train the full model to see overall performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[SELECTED_FEATURES])

    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, df[target_col])

    # 3. CALCULATE METRICS (Full Model)
    # ---------------------------------------------------------
    y_pred = model.predict(X_scaled)

    print("\nOverall Model Performance Metrics:")
    print("-" * 30)
    print(f"Accuracy:  {accuracy_score(df[target_col], y_pred):.4f}")
    print(f"Precision: {precision_score(df[target_col], y_pred):.4f}")
    print(f"Recall:    {recall_score(df[target_col], y_pred):.4f}")
    print(f"F1 Score:  {f1_score(df[target_col], y_pred):.4f}")
    print("-" * 30)
    print("\nFull Classification Report:")
    print(classification_report(df[target_col], y_pred))

    # 4. VISUALIZE INDIVIDUAL DECISION BOUNDARIES
    # ---------------------------------------------------------
    print("\nGenerating 20 separate univariate decision plots...")

    # Create a 5x4 grid of subplots (since there are 20 features)
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(24, 20))
    axes = axes.flatten()  # Flatten 2D array of axes to 1D for easy looping

    for i, feature_name in enumerate(SELECTED_FEATURES):
        ax = axes[i]

        # Extract just this feature and the target
        # We reshape (-1, 1) because sklearn expects 2D array for features
        X_single = df[[feature_name]].values
        y_single = df[target_col].values

        # Train a "mini" Logistic Regression on JUST this feature
        # to visualize its specific decision boundary
        clf = LogisticRegression()
        clf.fit(X_single, y_single)

        # Create a range of values for the X-axis (the feature)
        x_range = np.linspace(X_single.min(), X_single.max(), 300).reshape(-1, 1)

        # Predict probabilities (The Sigmoid Curve)
        y_prob = clf.predict_proba(x_range)[:, 1]

        # PLOT: Data points
        # We add a little random noise (jitter) to the y-values (0 and 1)
        # so the points don't all overlap, making them easier to see.
        jitter = np.random.normal(0, 0.02, size=len(y_single))
        ax.scatter(X_single, y_single + jitter, c=y_single, cmap='coolwarm', alpha=0.3, edgecolor='none', s=10)

        # PLOT: Sigmoid Curve
        ax.plot(x_range, y_prob, color='black', linewidth=2, label='Prob(Class=1)')

        # PLOT: Decision Boundary (where Probability = 0.5)
        # We find the x-value where y_prob crosses 0.5
        boundary_indices = np.where(np.abs(y_prob - 0.5) < 0.01)[0]
        if len(boundary_indices) > 0:
            boundary_x = x_range[boundary_indices[0]][0]
            ax.axvline(boundary_x, color='red', linestyle='--', label='Boundary (0.5)')

        # Styling
        ax.set_title(f"{feature_name[:20]}...", fontsize=10)  # Truncate long titles
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.2)

        # Only add legend to the first plot to avoid clutter
        if i == 0:
            ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.suptitle("Univariate Logistic Regression: Feature vs Target Probability", fontsize=20, y=1.02)
    plt.show()

    print("Plots generated successfully.")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def logistic_curves():
    target_col = 'class'
    # Assuming you have df with SELECTED_FEATURES and target_col defined

    # 1. SCALE THE FEATURES
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[SELECTED_FEATURES])

    # 2. REDUCE TO 2 DIMENSIONS WITH PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Explained variance: {pca.explained_variance_ratio_}")
    print(f"Total variance captured: {pca.explained_variance_ratio_.sum():.2%}")

    # 3. TRAIN LOGISTIC REGRESSION ON 2D PCA DATA
    model = LogisticRegression(random_state=42)
    model.fit(X_pca, df[target_col])

    # 4. CREATE DECISION BOUNDARY MESH
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict probabilities for each point in the mesh
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # 5. PLOT
    plt.figure(figsize=(12, 8))

    # Plot probability contours (sigmoid surface)
    contour = plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
    plt.colorbar(contour, label='Probability of Class 1')

    # Plot decision boundary (where probability = 0.5)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

    # Plot actual data points
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=df[target_col], cmap='RdYlBu_r',
                         edgecolor='black', s=50, alpha=0.8)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.title('Logistic Regression Decision Boundary (All Features via PCA)', fontsize=14)
    plt.legend(*scatter.legend_elements(), title="Class", loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print model performance
    y_pred = model.predict(X_pca)
    print(f"\nModel Accuracy on PCA data: {accuracy_score(df[target_col], y_pred):.4f}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def plot_individual_logistic_boundaries(df, features, target='class'):
    """
    Creates 20 separate plots, each showing one feature's decision boundary.
    """
    n_features = len(features)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]

        # Get feature and target
        X_single = df[[feature]].values
        y = df[target].values

        # Train logistic regression on this single feature
        model = LogisticRegression(random_state=42)
        model.fit(X_single, y)

        # Create mesh for decision boundary
        x_min, x_max = X_single.min() - 0.1, X_single.max() + 0.1
        xx = np.linspace(x_min, x_max, 300).reshape(-1, 1)

        # Get probabilities
        yy = model.predict_proba(xx)[:, 1]

        # Plot decision regions (background colors)
        # Create a 2D mesh for filled regions
        x_range = np.linspace(x_min, x_max, 300)
        y_range = np.linspace(-0.1, 1.1, 300)
        xx_mesh, yy_mesh = np.meshgrid(x_range, y_range)

        # Calculate probabilities for coloring
        prob_mesh = model.predict_proba(x_range.reshape(-1, 1))[:, 1]

        # Create background regions
        for i in range(len(x_range) - 1):
            if prob_mesh[i] < 0.5:
                color = 'lightblue'
            else:
                color = 'lightsalmon'
            ax.axvspan(x_range[i], x_range[i + 1], alpha=0.3, color=color)

        # Plot decision boundary line
        # Find where probability crosses 0.5
        boundary_idx = np.argmin(np.abs(yy - 0.5))
        boundary_x = xx[boundary_idx][0]
        ax.axvline(boundary_x, color='black', linestyle='--', linewidth=2,
                   label=f'Boundary (x={boundary_x:.2f})')

        # Plot data points
        class_0 = df[df[target] == 0]
        class_1 = df[df[target] == 1]

        # Add jitter to y-axis for visibility
        jitter_0 = np.random.normal(0, 0.02, len(class_0))
        jitter_1 = np.random.normal(1, 0.02, len(class_1))

        ax.scatter(class_0[feature], jitter_0, c='blue', alpha=0.5,
                   edgecolor='darkblue', s=30, label='Class 0')
        ax.scatter(class_1[feature], jitter_1, c='orange', alpha=0.5,
                   edgecolor='darkorange', s=30, label='Class 1')

        # Styling
        ax.set_xlabel(feature if len(feature) < 30 else feature[:27] + '...',
                      fontsize=9)
        ax.set_ylabel('Class', fontsize=9)
        ax.set_ylim(-0.15, 1.15)
        ax.set_yticks([0, 1])
        ax.set_title(f'{feature[:35]}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)

    # Hide extra subplots if any
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.suptitle('Individual Logistic Regression Decision Boundaries',
                 fontsize=16, y=1.01, fontweight='bold')
    plt.show()

selected_features = [
    'checking_status_no checking', 'checking_status_<0',
    'credit_history_critical/other existing credit',
    'age_(-0.001, 25.0]', 'duration_(36.0, 60.0]',
    'credit_amount_(10000.0, 20000.0]', 'checking_status_>=200',
    'purpose_new car', 'other_parties_guarantor',
    'duration_(24.0, 36.0]', 'savings_status_<100',
    'purpose_retraining', 'age_(60.0, 80.0]',
    'property_magnitude_real estate', 'savings_status_no known savings',
    'installment_commitment', 'credit_amount_(2500.0, 5000.0]',
    'employment_<1', 'property_magnitude_no known property',
    'credit_history_no credits/all paid'
]
# Usage
# plot_individual_logistic_boundaries(df, selected_features, target='class')

def plot_logistic_boundaries_with_sigmoid(df, features, target='class'):
    """
    Creates 20 plots with sigmoid probability curves and decision boundaries.
    """
    n_features = len(features)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]

        # Get feature and target
        X_single = df[[feature]].values
        y = df[target].values

        # Train logistic regression
        model = LogisticRegression(random_state=42)
        model.fit(X_single, y)

        # Create mesh
        x_min, x_max = X_single.min() - 0.1, X_single.max() + 0.1
        xx = np.linspace(x_min, x_max, 300).reshape(-1, 1)
        yy_prob = model.predict_proba(xx)[:, 1]

        # Create twin axis for probability curve
        ax2 = ax.twinx()

        # Plot probability curve (sigmoid)
        ax2.plot(xx, yy_prob, 'g-', linewidth=3, label='P(Class=1)', alpha=0.7)
        ax2.axhline(0.5, color='red', linestyle=':', linewidth=2,
                    label='Decision threshold')
        ax2.set_ylabel('Probability', fontsize=9, color='green')
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis='y', labelcolor='green')

        # Plot data points on main axis
        class_0 = df[df[target] == 0]
        class_1 = df[df[target] == 1]

        jitter_0 = np.random.normal(0, 0.02, len(class_0))
        jitter_1 = np.random.normal(1, 0.02, len(class_1))

        ax.scatter(class_0[feature], jitter_0, c='blue', alpha=0.4,
                   edgecolor='darkblue', s=30, label='Class 0', zorder=5)
        ax.scatter(class_1[feature], jitter_1, c='orange', alpha=0.4,
                   edgecolor='darkorange', s=30, label='Class 1', zorder=5)

        # Decision boundary
        boundary_idx = np.argmin(np.abs(yy_prob - 0.5))
        boundary_x = xx[boundary_idx][0]
        ax.axvline(boundary_x, color='black', linestyle='--', linewidth=2,
                   alpha=0.7, zorder=4)

        # Styling
        ax.set_xlabel(feature if len(feature) < 30 else feature[:27] + '...',
                      fontsize=9)
        ax.set_ylabel('Class', fontsize=9)
        ax.set_ylim(-0.15, 1.15)
        ax.set_yticks([0, 1])
        ax.set_title(f'{feature[:40]}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)

        # Add accuracy score
        y_pred = model.predict(X_single)
        accuracy = (y_pred == y).mean()
        ax.text(0.02, 0.98, f'Acc: {accuracy:.3f}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if idx == 0:
            ax.legend(loc='upper left', fontsize=7)
            ax2.legend(loc='upper right', fontsize=7)

    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.suptitle('Logistic Regression: Individual Feature Decision Boundaries with Sigmoid Curves',
                 fontsize=16, y=1.005, fontweight='bold')
    plt.show()


# Usage
plot_logistic_boundaries_with_sigmoid(df, selected_features, target='class')