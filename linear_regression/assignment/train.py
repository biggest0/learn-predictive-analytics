import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from imblearn.over_sampling import SMOTE
from datetime import datetime
import pickle
from itertools import combinations

SELECTED_FEATURES = ['checking_status_no checking', 'checking_status_<0',
                     'credit_history_critical/other existing credit',
                     'age_(-0.001, 25.0]', 'duration_(36.0, 60.0]', 'credit_amount_(10000.0, 20000.0]',
                     'checking_status_>=200', 'purpose_new car', 'other_parties_guarantor', 'duration_(24.0, 36.0]',
                     'savings_status_<100', 'purpose_retraining', 'age_(60.0, 80.0]', 'property_magnitude_real estate',
                     'savings_status_no known savings', 'installment_commitment', 'credit_amount_(2500.0, 5000.0]',
                     'employment_<1', 'property_magnitude_no known property', 'credit_history_no credits/all paid']

COLS_TO_DUMMY = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment',
                 'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing',
                 'job', 'own_telephone', 'foreign_worker']

COLS_TO_BIN = {
    "credit_amount": [0, 1500, 2500, 5000, 10000, 20000],
    "age": [0, 25, 35, 45, 60, 80],
    "duration": [0, 12, 18, 24, 36, 60],
}

PATH_TO_CREDIT_CSV = 'Credit_Train.csv'
PATH_TO_FULL_CLEANED_CSV = 'full_cleaned_data.csv'
PATH_TO_NUMERIC_CLEANED_CSV = 'numerical_cleaned_data.csv'


# ===============================================
# =============== LOAD/ SAVE DATA ===============
# ===============================================

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


# Save model with pickle
def save_model(model):
    with open(f"credit_model.pkl", "wb") as f:
        pickle.dump(model, f)


# Save model summary to txt
def save_model_summary_txt(summary, features):
    file_name = np.mean(summary['f1'])

    print(f'k_fold_{file_name:.4f}.txt')
    with open(f'k_fold_{file_name:.4f}.txt', 'w') as f:
        # Write metrics
        f.write("=== Metrics ===\n")
        for metric, values in summary.items():
            mean_value = np.mean(values)
            sd_value = np.std(values)
            f.write(f"{metric}: Mean = {mean_value:.4f}\n")
            f.write(f"{metric}: SD = {sd_value:.4f}\n")
        f.write("\n\n")

        # Write top 20 features
        f.write(f"=== Top {len(features)} Features ===\n")
        f.write('\n'.join(features))


# ===============================================
# ================= PRINT DATA ==================
# ===============================================

def print_kfold_summary_average(summary):
    for metric, values in summary.items():
        mean_value = np.mean(values)
        sd_value = np.std(values)
        print(f"{metric}: Mean = {mean_value:.4f}")
        print(f"{metric}: SD = {sd_value:.4f}")


def print_confusion_matrix(y_pred, y_test):
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print('Unseen confusion matrix')
    print(confusion_matrix)


def print_unseen_metrics(y_pred, y_test):
    f1 = metrics.f1_score(y_test, y_pred)
    print(f"Unseen F1: {f1}")


# ===============================================
# ============= FEATURE PROCESSING ==============
# ===============================================

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


# ===============================================
# ================ TRAIN MODEL ==================
# ===============================================

# Load df with selected features, target
def load_features_and_target(df, selected_features):
    X = df[selected_features]
    y = df['class']
    return X, y


def k_fold(X, y, NUM_SPLITS=5):
    metrics_summary = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    for _ in range(100):
        cv = KFold(n_splits=NUM_SPLITS, shuffle=True)
        for n, (train_indices, test_indicies) in enumerate(cv.split(X), start=1):
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indicies]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indicies]

            # Fit logistic regression model
            model = LogisticRegression(fit_intercept=True, solver='liblinear')
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

    return metrics_summary


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    logistic_model = LogisticRegression(fit_intercept=True, solver='liblinear')
    logistic_model.fit(X_train, y_train)

    return logistic_model, X_test, y_test


# ===============================================
# ============= SERVICE FUNCTIONS ===============
# ===============================================

# Clean data, process features, save copy to csv
def clean_dataframe():
    # Original uncleaned df
    df = get_csv_dataframe()

    df = knn_imputer(df)

    for col in COLS_TO_DUMMY:
        df = create_dummy_cols(df, col)

    for col, bin in COLS_TO_BIN.items():
        df = create_bin_cols(df, col, bin)

    save_csv_dataframe(df)

    return df


def output_model_summary(summary, model, X_test, y_test):
    y_pred = model.predict(X_test)

    print_kfold_summary_average(summary)

    print_confusion_matrix(y_pred, y_test)

    print_unseen_metrics(y_pred, y_test)


def train_and_evaluate_model(df):
    X, y = load_features_and_target(df, SELECTED_FEATURES)

    summary = k_fold(X, y)

    logistic_model, X_test, y_test = train_model(X, y)
    save_model(logistic_model)

    output_model_summary(summary, logistic_model, X_test, y_test)


# Driver
def main():
    df = clean_dataframe()

    train_and_evaluate_model(df)


if __name__ == '__main__':
    main()
