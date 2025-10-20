from numpy import array
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import precision_score

def basic_kfold():
    # data sample
    data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    # splits data into 3 randomized folds
    kfold = KFold(n_splits=3, shuffle=True)

    # enumerate splits
    for train, test in kfold.split(data):
        print('train: %s, test: %s' % (data[train], data[test]))
    # The output shows how three separate random folds


def actual_kfold(df):
    # prepare cross validation with three folds and 1 as a random seed.
    kfold = KFold(n_splits=3, shuffle=True)
    accuracyList = []
    foldCount = 0

    for trainIdx, testIdx in kfold.split(df):
        X_train, X_test, y_train, y_test = getTestAndTrainData(trainIdx, testIdx, df)

        # Recommended to only fit on training data.
        # Scaling only needed for X since y ranges between 0 and 1.
        scalerX = StandardScaler()
        X_train_scaled = scalerX.fit_transform(X_train)  # Fit and transform.
        X_test_scaled = scalerX.transform(X_test)  # Transform only.

        # Perform logistic regression.
        logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear')

        # Fit the model.
        logisticModel.fit(X_train_scaled, y_train)
        y_pred = logisticModel.predict(X_test_scaled)
        y_prob = logisticModel.predict_proba(X_test_scaled)
        # Show confusion matrix and accuracy scores.
        y_test_array = np.array(y_test['Purchased'])
        cm = pd.crosstab(y_test_array, y_pred, rownames=['Actual'], colnames=['Predicted'])
        print("\n***K-fold: " + str(foldCount))
        foldCount += 1
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracyList.append(accuracy)
        print('\nAccuracy: ', accuracy)
        print("\nConfusion Matrix")
        print(cm)

    print(classification_report(y_test, y_pred))
    precision = precision_score(y_test, y_pred)
    print('Precision: {0:0.2f}'.format(precision))

    auc = roc_auc_score(y_test, y_prob[:, 1], )
    print('Logistic: ROC AUC=%.3f' % (auc))
    print("\nAccuracy and Standard Deviation For All Folds:")
    print("*********************************************")
    print("Average accuracy: " + str(np.mean(accuracyList)))
    print("Accuracy std: " + str(np.std(accuracyList)))

def getTestAndTrainData(trainIndexes, testIndexes, df):
    dfTrain = df.iloc[trainIndexes, :] # Gets all rows with train indexes.
    dfTest = df.iloc[testIndexes, :]
    X_train = dfTrain[['EstimatedSalary', 'Age']]
    X_test = dfTest[['EstimatedSalary', 'Age']]
    y_train = dfTrain[['Purchased']]
    y_test = dfTest[['Purchased']]
    return X_train, X_test, y_train, y_test