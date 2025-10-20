import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, f1_score, roc_auc_score, \
    recall_score
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from file_handler import get_dataframe_with_features, get_csv_dataframe
from constant import TOP_FEATURES, MAYBE_FEATURES
from impute import convertNAcellsToNum
from create_dummy import create_dummy

def k_fold_example():
    # pd.set_option('display.max_columns', None)   # show all columns
    # pd.set_option('display.max_rows', None)      # show all rows (optional)
    # pd.set_option('display.width', None)         # let pandas decide based on your console
    # pd.set_option('display.expand_frame_repr', False)  # don't wrap to multiple lines
    # pd.read_csv(CSV_PATH, header = 0)
    df = get_csv_dataframe()
    # df = create_dummy(df, 'cancellation_policy')
    df = convertNAcellsToNum('bathrooms', df, "mean")
    df = convertNAcellsToNum('bedrooms', df, "mean")
    df = convertNAcellsToNum('beds', df, "mean")
    features = ['accommodates', 'imp_bedrooms', 'imp_bathrooms', 'imp_beds', 'indoor_fireplace', 'tv', 'dryer']

    X = df[features]
    X = sm.add_constant(X)
    y = df['price']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # if not doing k fold use this

    k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
    accuracy_list = []
    fold_count = 0
    r2_list, rmse_list, adjr2_list, fstat_list, f_pval_list, aic_list, bic_list = ([] for _ in range(7))

    for train_index, test_index in k_fold.split(X):
        fold_count += 1
        # get all rows with train indexes
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the OLS model
        model = sm.OLS(y_train, X_train).fit()

        # Predict on the test fold
        y_pred = model.predict(X_test)

        # Calculate R^2 score as "accuracy" for regression
        # r2 = r2_score(y_test, y_pred)
        # accuracy_list.append(r2)

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Store OLS summary stats
        r2_list.append(model.rsquared)
        adjr2_list.append(model.rsquared_adj)
        fstat_list.append(model.fvalue)
        f_pval_list.append(model.f_pvalue)
        aic_list.append(model.aic)
        bic_list.append(model.bic)
        rmse_list.append(rmse)

        # print(model.summary())
        print(f"Fold {fold_count}: R²={model.rsquared:.4f}, RMSE={rmse:.2f}, Adj R²_adj={model.rsquared_adj:.4f}, F={model.fvalue:.2f}")

    # print(f'\nAverage R² across all folds: {sum(accuracy_list) / len(accuracy_list):.4f}')
    print("\n===== Cross-Validation Averages =====")
    print(f"Average R²:       {np.mean(r2_list):.4f}")
    print(f"Average Adj R²:   {np.mean(adjr2_list):.4f}")
    print(f"Average RMSE:     {np.mean(rmse_list):.4f}")
    print(f"Average F-Stat:   {np.mean(fstat_list):.4f}")
    print(f"Average F p-val:  {np.mean(f_pval_list):.4e}")
    print(f"Average AIC:      {np.mean(aic_list):.4f}")
    print(f"Average BIC:      {np.mean(bic_list):.4f}")

k_fold_example()


def k_fold_logistic():
    df = get_csv_dataframe()
    # df = create_dummy(df, 'cancellation_policy')
    df = convertNAcellsToNum('bathrooms', df, "mean")
    df = convertNAcellsToNum('bedrooms', df, "mean")
    df = convertNAcellsToNum('beds', df, "mean")
    features = ['accommodates', 'imp_bedrooms', 'imp_bathrooms', 'imp_beds', 'indoor_fireplace', 'tv', 'dryer']

    X = df[features]
    y = df['some_binary_col']

    k_fold = KFold(n_splits=3, shuffle=True)
    accuracyList = []
    foldCount = 0

    for train_index, test_index in k_fold.split(X):
        foldCount += 1
        # get all rows with train indexes
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Recommended to only fit on training data.
        # Scaling only needed for X since y ranges between 0 and 1.
        scalerX = StandardScaler()
        X_train_scaled = scalerX.fit_transform(X_train)  # Fit and transform.
        X_test_scaled = scalerX.transform(X_test)  # Transform only.

        # Perform logistic regression, fit model (train)
        logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear')
        logisticModel.fit(X_train_scaled, y_train)

        y_pred = logisticModel.predict(X_test_scaled)
        y_prob = logisticModel.predict_proba(X_test_scaled)
        # Show confusion matrix and accuracy scores.
        y_test_array = np.array(y_test['Purchased'])
        cm = pd.crosstab(y_test_array, y_pred, rownames=['Actual'], colnames=['Predicted'])
        tn, fp, fn, tp = cm.ravel()

        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred) #recall
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob[:, 1], )

        accuracyList.append(accuracy)