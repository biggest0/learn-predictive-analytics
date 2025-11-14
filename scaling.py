"""
Examples of cross-validation and scaling techniques.
Uses modular functions from across the codebase.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Import from existing modules
from util.file_handler import get_csv_dataframe
from feature_creation.impute import convertNAcellsToNum
from feature_creation.dummy import create_dummy
from feature_interpretation.k_fold import run_kfold
from models.scaling import scaling_OLS, scale_and_train_ols

def k_fold_example():
    """
    Example of k-fold cross validation for regression using modular functions.
    """
    df = get_csv_dataframe()
    df = convertNAcellsToNum('bathrooms', df, "mean")
    df = convertNAcellsToNum('bedrooms', df, "mean")
    df = convertNAcellsToNum('beds', df, "mean")

    features = ['accommodates', 'imp_bedrooms', 'imp_bathrooms', 'imp_beds', 'indoor_fireplace', 'tv', 'dryer']
    X = df[features]
    X = sm.add_constant(X)
    y = df['price']

    # Use the modular k-fold function with comprehensive metrics
    print("Performing k-fold cross validation with comprehensive metrics...")
    results = run_kfold(X, y, n_splits=5, random_state=0, verbose=True, include_additional_metrics=True)

    print("\n===== Cross-Validation Summary =====")
    print(f"Average R²:     {results['mean_r2']:.4f}")
    print(f"Average Adj R²: {results['mean_adj_r2']:.4f}")
    print(f"Average RMSE:   {results['mean_rmse']:.4f}")
    print(f"Average AIC:    {results['mean_aic']:.4f}")
    print(f"Average BIC:    {results['mean_bic']:.4f}")

# k_fold_example()


def k_fold_logistic():
    """
    Example of k-fold cross validation for logistic regression.
    Note: This uses a simplified approach since the main run_kfold function is designed for OLS.
    """
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    df = get_csv_dataframe()
    df = convertNAcellsToNum('bathrooms', df, "mean")
    df = convertNAcellsToNum('bedrooms', df, "mean")
    df = convertNAcellsToNum('beds', df, "mean")

    features = ['accommodates', 'imp_bedrooms', 'imp_bathrooms', 'imp_beds', 'indoor_fireplace', 'tv', 'dryer']
    X = df[features]
    y = df['price'] > df['price'].median()  # Create binary target for demonstration

    k_fold = KFold(n_splits=3, shuffle=True, random_state=42)
    accuracy_list = []
    fold_count = 0

    for train_index, test_index in k_fold.split(X):
        fold_count += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train logistic regression
        model = LogisticRegression(fit_intercept=True, solver='liblinear')
        model.fit(X_train_scaled, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob[:, 1])

        accuracy_list.append(accuracy)
        print(f"Fold {fold_count}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
              f"Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    print(f"\nAverage Accuracy across {fold_count} folds: {np.mean(accuracy_list):.4f}")


def demo_scaling_ols():
    """
    Demo of scaling functionality using modular functions.
    """
    print("Running OLS with scaling demo...")
    scaling_OLS()


def demo_scaling_logistic():
    """
    Demo of logistic regression with scaling using modular functions.
    """
    from models.scaling import scaling_logistic
    print("Running logistic regression with scaling demo...")
    scaling_logistic()


def demo_advanced_scaling():
    """
    Demo of advanced scaling techniques using the modular scaling functions.
    """
    import pandas as pd
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # Load wine dataset
    wine = datasets.load_wine()
    dataset = pd.DataFrame(
        data=pd.c_[wine['data'], wine['target']],
        columns=wine['feature_names'] + ['target']
    )

    # Prepare data
    X = dataset.copy()
    X = X.drop(['target', 'hue', 'ash', 'magnesium', 'malic_acid', 'alcohol'], axis=1)
    y = dataset['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training OLS model with Robust scaling...")
    model, predictions, rmse = scale_and_train_ols(
        X_train, y_train, X_test, y_test,
        scaler_type='robust',
        save_scalers=True
    )

    print(f"Model trained successfully. RMSE: {rmse:.4f}")

    # Demo loading and predicting with saved scalers
    from models.scaling import load_scalers_and_predict
    print("\nTesting scaler loading and prediction...")
    predictions_loaded = load_scalers_and_predict(
        X_test[:5],  # Test on first 5 samples
        model_path='ols_model.pkl',
        scaler_x_path='scaler_X.pkl',
        scaler_y_path='scaler_y.pkl'
    )
    print(f"Loaded model predictions: {predictions_loaded.flatten()[:5]}")