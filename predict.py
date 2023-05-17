"""
Name: Rinji Le
Student ID: 13344552
Course: Afstudeerproject Bachelor Informatica
Date: TODO

Description:
TODO
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, LinearRegression, SGDRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.multiclass import OneVsRestClassifier  # TODO: OneVsOneClassifier, OutputCodeClassifier
from sklearn.model_selection import GridSearchCV


def check_file_exists(filename):
    """
    Checks if the file already exists. If it does, the user is asked to
    overwrite the file or not.

    :param filename: Name of the file (str).
    :return: None.
    """
    if os.path.isfile(f"data/{filename}.txt"):
        print("File already exists!")
        while True:
            answer = input("Do you want to overwrite the file? (y/n): ")
            if answer == "y":
                break
            elif answer == "n":
                sys.exit(0)


def my_accuracy_score(y_true, y_pred):
    correct = 0

    for i, row in y_pred.iterrows():
        if row.equals(y_true.iloc[i]):
            correct += 1

    return correct / y_pred.shape[0]


def prob2target(prob):
    for i, row in prob.iterrows():
        max_prob = max(row)

        if row.value_counts()[max_prob] > 1:
            max_indices = np.where(row == max_prob)[0]
            prob.iloc[i] = [0, 0, 0]
            prob.iloc[i][np.random.choice(max_indices)] = 1
        else:
            prob.iloc[i] = [1 if p == max_prob else 0 for p in row]

    return prob.astype(int)


def classification_prediction(data, targets, train_size=0.8, hyperparams_tuning=True):
    train_len = int(data.shape[0] * train_size)

    X_train = data.iloc[:train_len]
    X_test = data.iloc[train_len:]
    y_train = targets.iloc[:train_len]
    y_test = targets.iloc[train_len:]
    y_test.reset_index(drop=True, inplace=True)

    # TODO: params
    models = [
        (LogisticRegression(max_iter=10**4),  # To prevent ConvergenceWarning
         {
             "estimator__C": [0.1, 1, 10, 100],
             "estimator__solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
          }),
        (GaussianNB(),
         {}),
        (BernoulliNB(),
         {}),
        (MultinomialNB(),
         {}),
        (KNeighborsClassifier(),
         {}),
        (SGDClassifier(loss="log_loss"),
         {}),
        (SVC(probability=True),
         {}),
        (MLPClassifier(max_iter=10**4),
         {}),
        (DecisionTreeClassifier(),
         {}),
        (RandomForestClassifier(),
         {})
        # MultiOutputClassifier(GaussianProcessClassifier()),
        # RidgeClassifier(),
        # MultiOutputClassifier(ComplementNB()),
        # ExtraTreeClassifier(),
        # MultiOutputClassifier(AdaBoostClassifier()),
    ]

    accuracies = []

    for (model, params) in tqdm(models, desc="Predicting with the models"):
        if hyperparams_tuning:
            model = GridSearchCV(OneVsRestClassifier(model), params, cv=5, scoring="accuracy")
            model.fit(X_train, y_train)
            model = model.best_estimator_
        else:
            model = OneVsRestClassifier(model)
            model.fit(X_train, y_train)

        y_pred_prob = model.predict_proba(X_test)
        y_pred = prob2target(pd.DataFrame(y_pred_prob))

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append((model.estimator, accuracy))

    return accuracies


def classification(data):
    targets = pd.read_csv("data/ml_targets_result.csv", low_memory=False, header=None)
    # data = data.head(50)
    # targets = targets.head(50)

    accuracies = classification_prediction(data, targets, hyperparams_tuning=True)
    return sorted(accuracies, key=lambda x: x[1], reverse=True)


def regression_prediction(data, targets, train_size=0.8, hyperparams_tuning=True):
    train_len = int(data.shape[0] * train_size)

    X_train = data.iloc[:train_len]
    X_test = data.iloc[train_len:]
    y_train = targets.iloc[:train_len]
    y_test = targets.iloc[train_len:]
    y_test.reset_index(drop=True, inplace=True)

    # TODO: params
    models = [
        (LinearRegression(),
         {}),
        (SVR(),
         {}),
        (SGDRegressor(),
         {}),
        (GradientBoostingRegressor(),
         {})
    ]

    accuracies = []

    for (model, params) in tqdm(models, desc="Predicting with the models"):
        if hyperparams_tuning:
            model = GridSearchCV(model, params, cv=5, scoring="accuracy")
            model.fit(X_train, y_train)
            model = model.best_estimator_
        else:
            model = MultiOutputRegressor(model)
            model.fit(X_train, y_train)

        y_pred = pd.DataFrame(model.predict(X_test))

        # Round to the nearest integer
        y_pred = y_pred.applymap(lambda x: np.floor(x) if x % 1 < 0.5 else np.ceil(x)).astype(int)
        # Set negative values to 0
        y_pred = y_pred.applymap(lambda x: 0 if x < 0 else x)

        accuracy = my_accuracy_score(y_test, y_pred)
        accuracies.append((model.estimator, accuracy))

    return accuracies


def regression(data):
    targets = pd.read_csv("data/ml_targets_score.csv", low_memory=False, header=None)
    # data = data.head(500)
    # targets = targets.head(500)

    accuracies = regression_prediction(data, targets, hyperparams_tuning=True)
    return sorted(accuracies, key=lambda x: x[1], reverse=True)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 predict.py <file name>")
        sys.exit(1)

    filename = sys.argv[1]
    check_file_exists(filename)

    data = pd.read_csv("data/ml_data.csv", low_memory=False)
    classification_accuracies = classification(data)
    regression_accuracies = regression(data)

    with open(f"data/{filename}.txt", "w") as f:
        for (model, accuracy) in classification_accuracies:
            f.write(f"{model}: {accuracy * 100:.2f} %\n")

        print("\n")

        for (model, accuracy) in regression_accuracies:
            f.write(f"{model}: {accuracy * 100:.2f} %\n")


if __name__ == "__main__":
    main()
