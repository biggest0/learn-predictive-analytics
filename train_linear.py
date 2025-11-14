"""
Training pipeline for AirBNB price prediction using scikit-learn Linear Regression.
Alternative to statsmodels OLS approach.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Import from existing modules
from util.file_handler import get_csv_dataframe
from models.evaluation import evaluate_regression_model
from feature_creation.impute import knn_imputer
from feature_creation.dummy import create_dummy_cols, create_dummy_cols_datetime
from feature_creation.bin import create_bin_cols
from util.file_handler import save_local_model, load_local_model


def clean_data():
    """
    Complete data cleaning pipeline for AirBNB data.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for modeling
    """
    df = get_csv_dataframe()

    # Impute missing values
    df = knn_imputer(df)

    # Create dummy variables for categorical data
    cols_to_dummy = ['city', 'bed_type', 'room_type', 'cancellation_policy', 'cleaning_fee']
    for col in cols_to_dummy:
        df = create_dummy_cols(df, col)

    # Create dummy variables for datetime
    df = create_dummy_cols_datetime(df, 'host_since')

    # Create bins for review scores
    df = create_bin_cols(df, 'review_scores_rating', [0, 20, 40, 60, 80, 100])

    # Save cleaned data
    output_path = "cleaned_AirBNB_data_linear.csv"
    df.to_csv(output_path, index=False)

    return df


def create_test_split(df, features=None, target='price', test_size=0.15, random_state=0, scale_features=False):
    """
    Create train/test split for modeling with optional feature scaling.

    Parameters:
        df: DataFrame with features and target
        features: List of feature column names (optional, uses predefined if None)
        target: Target column name
        test_size: Proportion of test set
        random_state: Random seed
        scale_features: Whether to scale features using StandardScaler

    Returns:
        tuple: (X_train, X_test, y_train, y_test) or (X_train_scaled, X_test_scaled, y_train, y_test, scaler) if scaled
    """
    if features is None:
        # Default features for AirBNB model
        features = ['cancellation_policy_super_strict_60', 'bathrooms', 'smartlock',
                    'translation_missing:_en_hosting_amenity_49', 'indoor_fireplace', 'city_DC', 'doorman',
                    'room_type_Private room', 'city_NYC', 'cancellation_policy_strict', 'cable_tv', 'dryer',
                    'suitable_for_events', 'tv', 'family_kid_friendly', 'beds', 'bed_type_Couch',
                    'accommodates', 'city_SF', 'room_type_Shared room', 'city_Chicago', 'hot_tub',
                    'elevator', 'bedrooms']

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train, **model_kwargs):
    """
    Train a Linear Regression model.

    Parameters:
        X_train: Training features
        y_train: Training target
        **model_kwargs: Additional arguments for LinearRegression

    Returns:
        sklearn.linear_model.LinearRegression: Trained model
    """
    model = LinearRegression(**model_kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_linear_model(model, X_test, y_test, model_name="Linear Regression"):
    """
    Evaluate linear regression model performance.

    Parameters:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test target
        model_name: Name for display

    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} Evaluation:")
    print(f"R² (Coefficient of Determination): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    return {
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'predictions': y_pred
    }


def cross_validate_linear(X, y, cv=5, random_state=0, scoring='neg_mean_squared_error'):
    """
    Perform cross-validation for linear regression.

    Parameters:
        X: Feature matrix
        y: Target variable
        cv: Number of cross-validation folds
        random_state: Random seed
        scoring: Scoring metric for cross-validation

    Returns:
        dict: Cross-validation results
    """
    model = LinearRegression()

    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, random_state=random_state)

    if scoring == 'neg_mean_squared_error':
        # Convert negative MSE to positive RMSE
        rmse_scores = np.sqrt(-cv_scores)
        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)

        print(f"\nCross-Validation Results ({cv} folds):")
        print(f"RMSE: Mean = {mean_rmse:.4f}, SD = {std_rmse:.4f}")
        print(f"RMSE scores: {rmse_scores}")

        return {
            'rmse_scores': rmse_scores,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'cv_scores': cv_scores
        }
    else:
        # For other metrics
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)

        print(f"\nCross-Validation Results ({cv} folds):")
        print(f"Score: Mean = {mean_score:.4f}, SD = {std_score:.4f}")

        return {
            'cv_scores': cv_scores,
            'mean_score': mean_score,
            'std_score': std_score
        }


def plot_predictions_vs_actual(y_test, y_pred, title="Predictions vs Actual", save_path=None):
    """
    Plot predictions vs actual values.

    Parameters:
        y_test: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_residuals(y_test, y_pred, title="Residuals Plot", save_path=None):
    """
    Plot residuals (errors) vs actual values.

    Parameters:
        y_test: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save plot (optional)
    """
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals, alpha=0.6, edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_feature_importance(model, feature_names, title="Feature Importance", top_n=20, save_path=None):
    """
    Plot feature importance based on absolute coefficient values.

    Parameters:
        model: Trained LinearRegression model
        feature_names: List of feature names
        title: Plot title
        top_n: Number of top features to show
        save_path: Path to save plot (optional)
    """
    # Get coefficients and create importance DataFrame
    coefficients = model.coef_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(coefficients)
    })

    # Sort by importance and take top N
    importance_df = importance_df.sort_values('importance', ascending=True).tail(top_n)

    plt.figure(figsize=(10, 8))
    bars = plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Features')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Add coefficient values on bars
    for bar, value in zip(bars, importance_df['importance']):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                '.3f', ha='left', va='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def main():
    """
    Main training pipeline using scikit-learn Linear Regression.
    """
    print("🚀 Linear Regression Training Pipeline")
    print("=" * 50)

    # Clean and preprocess data
    print("📊 Loading and preprocessing data...")
    df = clean_data()
    print(f"Dataset shape: {df.shape}")

    # Create train/test split
    print("✂️  Creating train/test split...")
    X_train, X_test, y_train, y_test = create_test_split(df, scale_features=False)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Perform cross-validation
    print("🔄 Performing cross-validation...")
    cv_results = cross_validate_linear(X_train, y_train, cv=10, random_state=0)

    # Train final model
    print("🏃 Training final model...")
    model = train_linear_regression(X_train, y_train)

    # Evaluate on test set
    print("📈 Evaluating on test set...")
    eval_results = evaluate_linear_model(model, X_test, y_test, "Linear Regression")

    # Save model
    save_local_model(model, 'linear_regression_model')
    print("💾 Model saved as 'linear_regression_model_model.pkl'")

    # Create visualizations
    print("📊 Creating visualizations...")
    y_pred = eval_results['predictions']

    # Predictions vs Actual
    plot_predictions_vs_actual(y_test, y_pred,
                             title="Linear Regression: Predicted vs Actual Prices")

    # Residuals plot
    plot_residuals(y_test, y_pred,
                  title="Linear Regression: Residuals vs Actual Prices")

    # Feature importance
    feature_names = X_train.columns.tolist()
    plot_feature_importance(model, feature_names,
                          title="Linear Regression: Feature Importance (Top 20)",
                          top_n=20)

    print("\n✅ Linear Regression training completed!")
    print(f"📊 Final Model Performance:")
    print(f"   R² Score: {eval_results['r2']:.4f}")
    print(f"   RMSE: {eval_results['rmse']:.4f}")
    print(f"   MAE: {eval_results['mae']:.4f}")
if __name__ == '__main__':
    main()
