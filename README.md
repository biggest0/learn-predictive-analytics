# Airbnb Price Prediction - Predictive Analytics Project

A comprehensive machine learning project for predicting Airbnb listing prices using various regression techniques and feature engineering approaches.

## 🎯 Project Overview

This repository contains a complete predictive analytics pipeline for Airbnb price prediction, featuring:

- **Multiple Regression Approaches**: Statsmodels OLS and scikit-learn Linear Regression
- **Advanced Feature Engineering**: Imputation, categorical encoding, binning, outlier treatment
- **Model Evaluation**: Cross-validation, performance metrics, residual analysis
- **Visualization Suite**: Feature importance, prediction vs actual plots, correlation heatmaps
- **Modular Architecture**: Well-organized, reusable code structure

## 📁 Project Structure

```
├── data/                          # Data files and model outputs
│   ├── AirBNB.csv                # Main dataset
│   ├── AirBNB_mystery.csv        # Test data for predictions
│   └── ols_summary/              # Model summary reports
├── experiments/                  # Experimental and testing scripts
│   ├── modeling_experiments.py   # Feature selection and testing
│   ├── alternative_experiments.py # Alternative modeling approaches
│   └── plotting_demo.py          # Visualization examples
├── feature_creation/             # Feature engineering modules
│   ├── impute.py                 # Missing value imputation
│   ├── dummy.py                  # Categorical variable encoding
│   ├── bin.py                    # Numerical binning
│   └── outlier_treatment.py      # Outlier detection and treatment
├── feature_interpretation/       # Model interpretation tools
│   ├── k_fold.py                 # Cross-validation (regression & classification)
│   ├── feature_selection.py      # Feature selection algorithms
│   ├── graph.py                  # Visualization functions
│   └── interpret_features.py     # Feature analysis utilities
├── models/                       # Machine learning models
│   ├── training.py               # Model training utilities
│   ├── evaluation.py             # Model evaluation functions
│   ├── prediction.py             # Prediction and inference
│   └── scaling.py                # Feature scaling utilities
├── ols/                          # Original statsmodels approach
│   ├── train.py                  # OLS training pipeline
│   └── production.py             # Production deployment
├── util/                         # Utility functions
│   ├── file_handler.py           # Data loading and saving
│   └── util.py                   # General utilities
├── train.py                      # Main statsmodels training script
├── train_linear.py               # Main sklearn training script
└── README.md
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
```

### Basic Usage

**Statsmodels OLS Approach:**
```bash
python train.py
```

**Scikit-learn Linear Regression:**
```bash
python train_linear.py
```

## 🔧 Key Features

### Data Preprocessing Pipeline
```python
from feature_creation.impute import knn_imputer
from feature_creation.dummy import create_dummy_cols
from feature_creation.bin import create_bin_cols

# Complete preprocessing pipeline
df = knn_imputer(df)  # Handle missing values
df = create_dummy_cols(df, 'categorical_column')  # One-hot encoding
df = create_bin_cols(df, 'numeric_column', [0, 25, 50, 75, 100])  # Binning
```

### Model Training & Evaluation
```python
from models.training import train_ols_with_feature_selection
from models.evaluation import evaluate_regression_model
from feature_interpretation.k_fold import run_kfold

# Train with automatic feature selection and cross-validation
model, rmse, predictions = train_ols_with_feature_selection(
    features=['feature1', 'feature2', 'feature3'],
    run_kfold_cv=True,
    n_splits=10
)

# Evaluate performance
metrics = evaluate_regression_model(y_true, y_pred)
```

### Visualization Suite
```python
from feature_interpretation.graph import plot_scatter_vs_target, plot_correlation_heatmap

# Feature relationship visualization
fig, axes = plot_scatter_vs_target(df, ['feature1', 'feature2'], target='price')

# Correlation analysis
fig, ax = plot_correlation_heatmap(df, features, target='price')
```

## 📊 Available Models

### 1. Statsmodels OLS (`train.py`)
- **Approach**: Statistical linear regression with hypothesis testing
- **Features**: P-values, confidence intervals, AIC/BIC, F-statistics
- **Use Case**: Statistical analysis and inference
- **Output**: Detailed statistical summaries

### 2. Scikit-learn Linear Regression (`train_linear.py`)
- **Approach**: Machine learning linear regression
- **Features**: Cross-validation, feature importance, prediction intervals
- **Use Case**: Production deployment and prediction
- **Output**: RMSE, R², MAE, coefficient analysis

## 🔍 Feature Engineering Modules

### Imputation
```python
from feature_creation.impute import convertNAcellsToNum, KNN_imputer

# Mean imputation with indicators
df = convertNAcellsToNum('column_name', df, 'mean')

# KNN imputation for all numeric columns
df = KNN_imputer(df)
```

### Categorical Encoding
```python
from feature_creation.dummy import create_dummy_cols, create_dummy_cols_datetime

# Standard dummy variables
df = create_dummy_cols(df, 'categorical_column')

# Date-based encoding
df = create_dummy_cols_datetime(df, 'date_column')
```

### Feature Selection
```python
from feature_interpretation.feature_selection import forward_feature_selection, recursive_feature_elimination

# Forward selection
selected_features = forward_feature_selection(df, target='price')

# Recursive elimination
selected_features = recursive_feature_elimination(df, target='price')
```

## 📈 Cross-Validation

### Regression CV
```python
from feature_interpretation.k_fold import run_kfold, kfold_with_detailed_metrics

# Basic CV with RMSE
results = run_kfold(X, y, n_splits=5)

# Detailed CV with R², AIC, BIC
results = kfold_with_detailed_metrics(X, y, n_splits=5)
```

### Classification CV
```python
from feature_interpretation.k_fold import run_kfold_classification

# Classification cross-validation
results = run_kfold_classification(X, y, n_splits=5, scale_features=True)
```

## 🎨 Visualization Examples

```python
# Experiment with different plots
from experiments.plotting_demo import demo_scatter_plots, demo_heatmap_plots

demo_scatter_plots()  # Feature vs target scatter plots
demo_heatmap_plots()  # Correlation heatmaps
```

## 🔧 Advanced Usage

### Custom Model Training
```python
from models.training import prepare_data_for_modeling
from models.evaluation import save_model_and_summary

# Prepare data
df_numeric = prepare_data_for_modeling()

# Train custom model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate and save
metrics = evaluate_regression_model(y_test, model.predict(X_test))
```

### Production Deployment
```python
from ols.production import load_and_predict
from models.prediction import predict_mystery_data

# Load trained model and make predictions
predictions = load_and_predict(new_data)

# Or use the prediction pipeline
predictions = predict_mystery_data(model_path='path/to/model.pkl')
```

## 📋 Development Notes

### Recent Refactoring
- **Modularized codebase** into logical packages (`models/`, `feature_creation/`, etc.)
- **Unified cross-validation** functions for both regression and classification
- **Enhanced visualization** suite with customizable plotting functions
- **Standardized file handling** with consistent data loading/saving

### Key Improvements
- **DRY Principle**: Eliminated code duplication across files
- **Separation of Concerns**: Clear boundaries between data prep, modeling, and evaluation
- **Reusability**: Functions can be imported and used across different scripts
- **Maintainability**: Well-documented, modular code structure

## 🤝 Contributing

This is a learning project demonstrating various approaches to predictive analytics. The modular structure makes it easy to:

- Add new feature engineering techniques
- Implement additional ML algorithms
- Extend visualization capabilities
- Experiment with different evaluation metrics

## 📚 Learning Objectives Covered

- **Data Preprocessing**: Handling missing values, categorical encoding, feature scaling
- **Feature Engineering**: Creating meaningful features from raw data
- **Model Development**: Comparing statistical vs ML approaches
- **Model Evaluation**: Cross-validation, performance metrics, residual analysis
- **Visualization**: Effective communication of insights and results
- **Production Deployment**: Model persistence and inference pipelines

---

*Built for learning predictive analytics through hands-on implementation of end-to-end ML pipelines.*