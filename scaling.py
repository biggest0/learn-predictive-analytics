import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, f1_score, roc_auc_score, \
    recall_score, accuracy_score, confusion_matrix
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler

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

# k_fold_example()


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


def scaling_OLS():
    import pandas as pd
    import numpy as np
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm
    import numpy as np
    from sklearn import metrics

    wine = datasets.load_wine()
    dataset = pd.DataFrame(
        data=np.c_[wine['data'], wine['target']],
        columns=wine['feature_names'] + ['target']
    )

    # Create copy to prevent overwrite.
    X = dataset.copy()
    del X['target']  # Remove target variable
    del X['hue']  # Remove unwanted features
    del X['ash']
    del X['magnesium']
    del X['malic_acid']
    del X['alcohol']

    y = dataset['target']

    # Adding an intercept *** This is requried ***. Don't forget this step.
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    from sklearn.preprocessing import RobustScaler

    sc_x = RobustScaler()
    X_train_scaled = sc_x.fit_transform(X_train)

    # Create y scaler. Only scale y_train since evaluation
    # will use the actual size y_test.
    sc_y = RobustScaler()
    y_train_scaled = sc_y.fit_transform(np.array(y_train).reshape(-1, 1))

    # Save the fitted scalers.
    from pickle import dump, load

    dump(sc_x, open('sc_x.pkl', 'wb'))
    dump(sc_y, open('sc_y.pkl', 'wb'))

    # Build model with training data.
    model = sm.OLS(y_train_scaled, X_train_scaled).fit()

    # Save model
    dump(model, open('ols_model.pkl', 'wb'))

    # Load model
    loaded_model = load(open('ols_model.pkl', 'rb'))

    # Load the scalers.
    loaded_scalerX = load(open('sc_x.pkl', 'rb'))
    loaded_scalery = load(open('sc_y.pkl', 'rb'))

    X_test_scaled = loaded_scalerX.transform(X_test)
    scaledPredictions = loaded_model.predict(X_test_scaled)  # make predictions

    # Rescale predictions back to actual size range.
    predictions = loaded_scalery.inverse_transform(
        np.array(scaledPredictions).reshape(-1, 1))

    print(loaded_model.summary())
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, predictions)))

scaling_OLS()

def test():
    wine = datasets.load_wine()
    dataset = pd.DataFrame(
        data=np.c_[wine['data'], wine['target']],
        columns=wine['feature_names'] + ['target']
    )

    # Create copy to prevent overwrite.
    X = dataset.copy()
    del X['target']  # Remove target variable
    del X['hue']  # Remove unwanted features
    del X['ash']
    del X['magnesium']
    del X['malic_acid']
    del X['alcohol']

    y = dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler_x = RobustScaler()
    scaler_y = RobustScaler()
    # 1. SCALE
    X_train_scaled = scaler_x.fit_transform(X_train)  # Learn & scale training
    y_train_scaled = scaler_y.fit_transform(y_train)  # Learn & scale training y
    # Don't scale y_test - keep it original

    # 2. TRAIN & PREDICT
    model = sm.OLS(y_train_scaled, X_train_scaled).fit()
    scaledPredictions = model.predict(X_test)  # Predictions are in scaled units

    # 3. RESCALE & COMPARE
    predictions = scaler_y.inverse_transform(scaledPredictions)  # Back to original units
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))  # Compare original vs original

    dump(sc_x, open('sc_x.pkl', 'wb'))
    dump(sc_y, open('sc_y.pkl', 'wb'))
    dump(model, open('ols_model.pkl', 'wb'))     # Save model
    loaded_model = load(open('ols_model.pkl', 'rb'))     # Load model
    loaded_scalerX = load(open('sc_x.pkl', 'rb'))     # Load the scalers.
    loaded_scalery = load(open('sc_y.pkl', 'rb'))
    X_test_scaled = loaded_scalerX.transform(X_test)
    scaledPredictions = loaded_model.predict(X_test_scaled)  # make predictions

def scaling_logistic():
    # --- Example dataset ---
    df = pd.DataFrame({
        'income': [25, 30, 35, 40, 100, 120, 150, 200],
        'age': [20, 25, 30, 35, 40, 45, 50, 55],
        'bought': [0, 0, 0, 1, 1, 1, 1, 1]
    })

    # Features and target
    X = df[['income', 'age']]
    y = df['bought']

    # --- 1. Scale features ---
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 2. Split data ---
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

    # --- 3. Train logistic regression ---
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)

    # --- 4. Predict and evaluate ---
    y_pred = model.predict(X_test)

    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))