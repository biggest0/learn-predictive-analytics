import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from feature_interpretation.feature_selection import chi_square_feature_selection, \
    logistic_recursive_feature_elimination, anova_feature_selection, mutual_information_selection, \
    random_forest_importance, ensemble_feature_selection
from linear_regression.test.constants import CATEGORICAL_COLS_TO_DUMMY, COLS_TO_BIN, TOP_20_FEATURES

PATH_TO_CREDIT_CSV = 'Credit_Train.csv'
PATH_TO_FULL_CLEANED_CSV = 'full_cleaned_data.csv'
PATH_TO_NUMERIC_CLEANED_CSV = 'numerical_cleaned_data.csv'


# Get dataframe
def get_csv_dataframe(csv_path=PATH_TO_CREDIT_CSV):
    # show all columns
    pd.set_option('display.max_columns', None)
    # show all rows
    pd.set_option('display.max_rows', None)
    # let pandas decide based on your console
    pd.set_option('display.width', None)
    # don't wrap to multiple lines
    pd.set_option('display.expand_frame_repr', False)
    return pd.read_csv(csv_path, header=0)


# Save df to csv
def save_csv_dataframe(df):
    df.to_csv(PATH_TO_FULL_CLEANED_CSV, index=False)
    df_numeric = df.select_dtypes(include=['number'])
    df_numeric.to_csv(PATH_TO_NUMERIC_CLEANED_CSV, index=False)


# Imput missing data
def knn_imputer(df):
    # Create the imputer
    imputer = KNNImputer(n_neighbors=5)

    # Select only numeric columns for imputation
    numeric_df = df.select_dtypes(include=['number'])

    # Fit + transform the numeric columns
    imputed_array = imputer.fit_transform(numeric_df)

    # Put the imputed data back into a DataFrame
    imputed_df = pd.DataFrame(imputed_array, columns=numeric_df.columns)

    # Combine imputed numeric columns wit categorical columns
    categorical_df = df.select_dtypes(exclude='number')
    df = pd.concat([imputed_df, categorical_df], axis=1)

    return df


# Dummy variables
def create_dummy_cols(df, col_name):
    # Create dummy variables with 0/1 row value, drop first column
    dummy_df = pd.get_dummies(df[col_name], prefix=col_name, drop_first=True, dtype=int)

    # Join back to original df and drop the original column
    df_with_dummies = pd.concat(([df, dummy_df]), axis=1)

    return df_with_dummies


# Binning
def create_bin_cols(df, col_name, bins, drop_original=False):
    # Bin the df column
    binned = pd.cut(df[col_name], bins=bins, include_lowest=True)

    # Create dummies for each bin (0/1)
    dummy_binned_df = pd.get_dummies(binned, prefix=col_name, dtype=int)

    # Join with original DataFrame
    df_with_bins = pd.concat(([df, dummy_binned_df]), axis=1)

    # Optionally drop original column
    if drop_original:
        df_with_bins = df_with_bins.drop(columns=[col_name])

    return df_with_bins


def clean_dataframe():
    df = get_csv_dataframe()

    df = knn_imputer(df)

    for col in CATEGORICAL_COLS_TO_DUMMY:
        df = create_dummy_cols(df, col)

    for col, bin in COLS_TO_BIN.items():
        df = create_bin_cols(df, col, bin)

    save_csv_dataframe(df)


def feature_selection():
    df = get_csv_dataframe(PATH_TO_NUMERIC_CLEANED_CSV)

    rfe_features = set(logistic_recursive_feature_elimination(df))
    chi_features = set(chi_square_feature_selection(df))
    anova_features = anova_feature_selection(df)
    mutual_features = mutual_information_selection(df)
    forest_features = random_forest_importance(df)
    ensemble_features = ensemble_feature_selection(df)

    print('anova')
    print(anova_features, '\n')

    print('mutual')
    print(mutual_features, '\n')

    print('random forest')
    print(forest_features, '\n')

    print('ensemble')
    print(ensemble_features, '\n')

    print('rfe')
    print(rfe_features, '\n')

    print('chi')
    print(chi_features, '\n')


def save_model_results_txt(file_name, y_test, y_pred):
    with open(f'../data/models/{file_name}.txt', 'w') as f:
        # Write confusion matrix
        f.write("=== Confusion Matrix ===\n")
        confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        f.write(confusion_matrix.to_string())
        f.write("\n\n")

        # Write metrics
        f.write("=== Metrics ===\n")
        f.write(f"Accuracy:  {metrics.accuracy_score(y_test, y_pred):.4f}\n")
        f.write(f"Precision: {metrics.precision_score(y_test, y_pred):.4f}\n")
        f.write(f"Recall:    {metrics.recall_score(y_test, y_pred):.4f}\n")
        f.write(f"F1-score:  {metrics.f1_score(y_test, y_pred):.4f}\n")
        f.write("\n")

        # Write top 20 features
        f.write("=== Top 20 Features ===\n")
        f.write('\n'.join(TOP_20_FEATURES))


def save_k_fold_txt(summary, features):
    file_name = np.mean(summary['accuracy'])

    with open(f'../data/models/k_fold_{file_name:.4f}.txt', 'w') as f:
        # Write metrics
        f.write("=== Metrics ===\n")
        for metric, values in summary.items():
            mean_value = np.mean(values)
            f.write(f"{metric}: Mean = {mean_value:.4f}\n")
        f.write("\n\n")

        # Write top 20 features
        f.write("=== Top 20 Features ===\n")
        f.write('\n'.join(features))


def k_fold(X, y, NUM_SPLITS=5):
    cv = KFold(n_splits=NUM_SPLITS, shuffle=True)
    metrics_summary = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    for n, (train_indices, test_indicies) in enumerate(cv.split(X), start=1):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indicies]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indicies]

        # Fit logistic regression model
        model = LogisticRegression(fit_intercept=True, solver='liblinear', random_state=0)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate scores
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)

        # Store scores
        metrics_summary["accuracy"].append(accuracy)
        metrics_summary["precision"].append(precision)
        metrics_summary["recall"].append(recall)
        metrics_summary["f1"].append(f1)

        print(f"Fold {n} | accuracy: {accuracy:.4f} | precision: {precision:.4f} | "
              f"recall: {recall:.4f} | f1: {f1:.4f}")

    for metric, values in metrics_summary.items():
        mean_value = np.mean(values)
        print(f"{metric}: Mean = {mean_value:.4f}")

    return metrics_summary


df = get_csv_dataframe(PATH_TO_FULL_CLEANED_CSV)

X = df[TOP_20_FEATURES]
y = df['class']

summary = k_fold(X, y, 10)

save_k_fold_txt(summary, TOP_20_FEATURES)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
#
# logistic_model = LogisticRegression(fit_intercept=True, solver='liblinear', random_state=0)
# logistic_model.fit(X_train, y_train)
#
# y_pred = logistic_model.predict(X_test)
# print(y_pred)
#
# confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
# print(confusion_matrix)
# print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
# print('Precision:', metrics.precision_score(y_test, y_pred))
# print('Recall:', metrics.recall_score(y_test, y_pred))
# print('F1-score:', metrics.f1_score(y_test, y_pred))
#
# file_name = 'model_1'


# print(df.head(10))
# print(df.describe())
# print(df.info())
