import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_regression, RFE, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression


def forward_feature_selection(df):
    X = df.copy()  # Create separate copy to prevent unwanted tampering of data.
    del X['price']  # Delete target variable.

    # Target variable
    y = df['price']
    #  f_regression returns F statistic for each feature.
    ffs = f_regression(X, y)

    featuresDf = pd.DataFrame()
    for i in range(0, len(X.columns)):
        featuresDf = featuresDf._append({"feature": X.columns[i],
                                         "ffs": ffs[0][i]}, ignore_index=True)
    featuresDf = featuresDf.sort_values(by=['ffs'], ascending=False)

    # Get top 20 features
    top_20_features = featuresDf.head(20)
    print(top_20_features)
    top_features_list = top_20_features['feature'].tolist()
    print(top_features_list)


def recursive_feature_elimination(df):
    # Seperate the target and independent variable
    X = df.copy()  # Create separate copy to prevent unwanted tampering of data.
    del X['price']  # Delete target variable.

    # Target variable
    y = df['price']

    # Create the object of the model
    model = LinearRegression()

    # Specify the number of  features to select
    rfe = RFE(model, n_features_to_select=20)

    # fit the model
    rfe = rfe.fit(X, y)
    # Please uncomment the following lines to see the result
    print('\n\nFEATUERS SELECTED\n\n')
    print(rfe.support_)

    top_features_list = []
    columns = list(X.keys())
    for i in range(0, len(columns)):
        if (rfe.support_[i]):
            print(columns[i])
            top_features_list.append(columns[i])

    print(top_features_list)

# df = get_pkl_dataframe()
# forward_feature_selection(df)
# recursive_feature_elimination(df)


def logistic_forward_feature_selection(df, target_col='class', k=20):
    """
    Forward Feature Selection using F-statistic.
    Starts with no features and adds the best one at each step.
    """
    from sklearn.feature_selection import f_classif

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Calculate F-statistic for each feature
    f_scores, p_values = f_classif(X, y)

    # Create results dataframe
    features_df = pd.DataFrame({
        'feature': X.columns,
        'f_score': f_scores,
        'p_value': p_values
    })

    # Sort by F-score (higher is better)
    features_df = features_df.sort_values(by='f_score', ascending=False)

    # Get top k features
    selected_features = features_df.head(k)['feature'].tolist()

    print("=" * 80)
    print(f"FORWARD FEATURE SELECTION: Top {k} Features")
    print("=" * 80)
    print(features_df.head(k).to_string(index=False))
    print("\n")

    return selected_features


def logistic_recursive_feature_elimination(df, target_col='class', k=20):
    """
    Use RFE with Logistic Regression to select features.
    Recursively removes least important features.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Create the model
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Specify the number of features to select
    rfe = RFE(model, n_features_to_select=k)

    # Fit the model
    rfe.fit(X, y)

    # Get selected features
    selected_features = X.columns[rfe.support_].tolist()

    # Get feature rankings (1 = selected, >1 = not selected, lower rank = better)
    feature_rankings = pd.DataFrame({
        'feature': X.columns,
        'ranking': rfe.ranking_,
        'selected': rfe.support_
    })
    feature_rankings = feature_rankings.sort_values(by='ranking')

    print("=" * 80)
    print(f"RECURSIVE FEATURE ELIMINATION: Top {k} Features")
    print("=" * 80)
    print(feature_rankings.head(k).to_string(index=False))
    print("\n")

    return selected_features


def chi_square_feature_selection(df, target_col='class', k=20):
    """
    Use Chi-square test to select best features.
    NOTE: All features must be non-negative. Works best for categorical features.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Chi-square requires non-negative values
    # If you have negative values, you need to normalize/scale first
    if (X < 0).any().any():
        print("WARNING: Chi-square requires non-negative features.")
        print("Applying min-max scaling to make all values non-negative...\n")
        X = (X - X.min()) / (X.max() - X.min())

    # Calculate chi-square statistic
    chi_scores, p_values = chi2(X, y)

    # Create results dataframe
    features_df = pd.DataFrame({
        'feature': X.columns,
        'chi2_score': chi_scores,
        'p_value': p_values
    })

    # Sort by chi-square score (higher is better)
    features_df = features_df.sort_values(by='chi2_score', ascending=False)

    # Get top k features
    top_features = features_df.head(k)

    print("=" * 80)
    print(f"CHI-SQUARE TEST: Top {k} Features")
    print("=" * 80)
    print(top_features.to_string(index=False))
    print("\n")

    return top_features['feature'].tolist()


def anova_feature_selection(df, target_col='class', k=20):
    """
    Use ANOVA F-statistic to select best features for classification.
    Works well for numeric features predicting categorical target.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Calculate F-statistic for each feature
    f_scores, p_values = f_classif(X, y)

    # Create results dataframe
    features_df = pd.DataFrame({
        'feature': X.columns,
        'f_score': f_scores,
        'p_value': p_values
    })

    # Sort by F-score (higher is better)
    features_df = features_df.sort_values(by='f_score', ascending=False)

    # Get top k features
    top_features = features_df.head(k)

    print("=" * 80)
    print(f"ANOVA F-TEST: Top {k} Features")
    print("=" * 80)
    print(top_features.to_string(index=False))
    print("\n")

    return top_features['feature'].tolist()


def mutual_information_selection(df, target_col='class', k=20):
    """
    Use Mutual Information to select features.
    Captures both linear and non-linear relationships.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # Create results dataframe
    features_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    })

    # Sort by MI score (higher is better)
    features_df = features_df.sort_values(by='mi_score', ascending=False)

    # Get top k features
    top_features = features_df.head(k)

    print("=" * 80)
    print(f"MUTUAL INFORMATION: Top {k} Features")
    print("=" * 80)
    print(top_features.to_string(index=False))
    print("\n")

    return top_features['feature'].tolist()


def random_forest_importance(df, target_col='class', k=20):
    """
    Use Random Forest feature importance to select features.
    Accounts for non-linear relationships and feature interactions.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Create and train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Get feature importances
    importances = rf.feature_importances_

    # Create results dataframe
    features_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    })

    # Sort by importance (higher is better)
    features_df = features_df.sort_values(by='importance', ascending=False)

    # Get top k features
    top_features = features_df.head(k)

    print("=" * 80)
    print(f"RANDOM FOREST IMPORTANCE: Top {k} Features")
    print("=" * 80)
    print(top_features.to_string(index=False))
    print("\n")

    return top_features['feature'].tolist()


def ensemble_feature_selection(df, target_col='class', k=20):
    """
    Combine multiple feature selection methods and select features
    that appear most frequently across methods.
    """
    print("=" * 80)
    print("RUNNING ENSEMBLE FEATURE SELECTION")
    print("=" * 80)
    print("\n")

    # Run all methods
    anova_features = set(anova_feature_selection(df, target_col, k))
    chi2_features = set(chi_square_feature_selection(df, target_col, k))
    mi_features = set(mutual_information_selection(df, target_col, k))
    rfe_features = set(logistic_recursive_feature_elimination(df, target_col, k))
    rf_features = set(random_forest_importance(df, target_col, k))

    # Count how many times each feature appears
    all_features = (
            list(anova_features) + list(chi2_features) +
            list(mi_features) + list(rfe_features) + list(rf_features)
    )

    feature_counts = pd.Series(all_features).value_counts()

    # Create results dataframe
    ensemble_df = pd.DataFrame({
        'feature': feature_counts.index,
        'method_count': feature_counts.values,
        'in_anova': feature_counts.index.isin(anova_features),
        'in_chi2': feature_counts.index.isin(chi2_features),
        'in_mi': feature_counts.index.isin(mi_features),
        'in_rfe': feature_counts.index.isin(rfe_features),
        'in_rf': feature_counts.index.isin(rf_features),
    })

    ensemble_df = ensemble_df.sort_values(by='method_count', ascending=False)

    print("=" * 80)
    print(f"ENSEMBLE RESULTS: Features Selected by Multiple Methods")
    print("=" * 80)
    print(ensemble_df.head(30).to_string(index=False))
    print("\n")

    # Return top k features that appear in most methods
    top_ensemble_features = ensemble_df.head(k)['feature'].tolist()

    return top_ensemble_features