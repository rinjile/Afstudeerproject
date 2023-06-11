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
import time
import datetime
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, learning_curve

random_seed = 10
np.random.seed(random_seed)


def check_file_exists(filename):
    """
    Checks if the file already exists. If it does, the user is asked to
    overwrite the file or not.

    :param filename: Name of the file (str).
    :return: None.
    """
    if os.path.isfile(f"results/{filename}.csv"):
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


def prob2target(prob,  t):
    for i, row in prob.iterrows():
        max_prob = max(row)

        if max_prob < t:
            prob.iloc[i] = [-1, -1, -1]
            continue

        if row.value_counts()[max_prob] > 1:
            max_indices = np.where(row == max_prob)[0]
            prob.iloc[i] = [0, 0, 0]
            prob.iloc[i][np.random.choice(max_indices)] = 1
        else:
            prob.iloc[i] = [1 if p == max_prob else 0 for p in row]

    return prob.astype(int)


def save_learning_curve(model, learning_curve_params, n, ci=95):
    train_sizes, train_scores, validation_scores = learning_curve(
        model, **learning_curve_params)
    train_ci_lower = np.percentile(train_scores, (100 - ci) / 2, axis=1)
    train_ci_upper = np.percentile(train_scores, (100 + ci) / 2, axis=1)
    validation_ci_lower = np.percentile(validation_scores, (100 - ci) / 2,
                                        axis=1)
    validation_ci_upper = np.percentile(validation_scores, (100 + ci) / 2,
                                        axis=1)

    with open(f"results/n{n}/learning_curve_"
              f"{model.estimator.__class__.__name__}{n}.csv", "w") as f:
        f.write("train_size,train_mean,train_ci_lower,train_ci_upper,"
                "validation_mean,validation_ci_lower,validation_ci_upper\n")

        for i in range(train_sizes.shape[0]):
            f.write(f"{train_sizes[i]}," f"{train_scores[i].mean()},"
                    f"{train_ci_lower[i]}," f"{train_ci_upper[i]},"
                    f"{validation_scores[i].mean()},"
                    f"{validation_ci_lower[i]},"
                    f"{validation_ci_upper[i]}\n")


def save_accuracies(classification_accuracies, regression_accuracies, n,
                    filename):
    with open(f"results/n{n}/{filename}{n}.csv", "w") as f:
        f.write("type,model,accuracy,hyperparameters\n")

        for (model, accuracy) in classification_accuracies:
            f.write(f"classification,{model.__class__.__name__},"
                    f"{accuracy * 100:.2f},\"{model.get_params()}\"\n")

        for (model, accuracy) in regression_accuracies:
            f.write(f"regression,{model.__class__.__name__},"
                    f"{accuracy * 100:.2f},\"{model.get_params()}\"\n")


def save_execution_times(start_time, start_time_str, classification_time, n):
    with open(f"results/n{n}/execution_time{n}.txt", "w") as f:
        f.write(f"Started at: {start_time_str}.\n")
        f.write(f"Ended at: "
                f"{datetime.datetime.now().strftime('%d-%m-%Y %H:%M')}.\n\n")
        f.write(f"Classification time: "
                f"{((classification_time - start_time) / 3600):.2f} hours.\n")
        f.write(f"Regression time: "
                f"{((time.time() - classification_time) / 3600):.2f} hours.\n")
        f.write(f"Total execution time: "
                f"{((time.time() - start_time) / 3600):.2f} hours.\n")


def classification_prediction(data, targets, hyperparams_tuning, n, verbose, t,
                              train_size=0.8):
    train_len = int(data.shape[0] * train_size)

    X_train = data.iloc[:train_len]
    X_test = data.iloc[train_len:]
    y_train = targets.iloc[:train_len]
    y_test = targets.iloc[train_len:]
    y_test.reset_index(drop=True, inplace=True)

    models = [
        (LogisticRegression(random_state=random_seed),
         {
             "estimator__C": [0.1, 0.5, 1, 3],
             "estimator__solver": ["lbfgs", "liblinear", "newton-cg",
                                   "newton-cholesky", "sag", "saga"]
         }),
        (GaussianNB(),
         {}),  # No hyperparameters to tune
        (BernoulliNB(),  # (TODO: deze doen?)
         {
                "estimator__alpha": [0.01, 0.1, 0.5, 1, 2],
         }),
        (MultinomialNB(),
         {
             "estimator__alpha": [0.01, 0.1, 0.5, 1, 2],
             "estimator__fit_prior": [True, False],
             "estimator__class_prior": [None, [0.2, 0.8], [0.3, 0.7],
                                        [0.4, 0.6], [0.5, 0.5]]
         }),
        (KNeighborsClassifier(),
         {
             "estimator__n_neighbors": range(1, 11),
             "estimator__weights": ["uniform", "distance"],
             "estimator__p": [1, 2, 3]
         }),
        (SVC(probability=True, max_iter=1000, random_state=random_seed),
         {
             "estimator__kernel": ["linear", "poly", "rbf", "sigmoid"],
             "estimator__gamma": ["scale", "auto"]
         }),
        (MLPClassifier(random_state=random_seed),
         {
             "estimator__hidden_layer_sizes": [(50,), (100,), (500,), (50, 2),
                                               (100, 2), (500, 2), (50, 3),
                                               (100, 3), (500, 3)],
             "estimator__activation": ["identity", "logistic", "tanh", "relu"],
             "estimator__alpha": [0.0001, 0.001, 0.01, 0.1, 1],
         }),
        (DecisionTreeClassifier(random_state=random_seed),
         {
             "estimator__criterion": ["gini", "entropy", "log_loss"],
             "estimator__splitter": ["best", "random"],
             "estimator__max_features": [None, "sqrt", "log2"],
         }),
        (RandomForestClassifier(random_state=random_seed),
         {
             "estimator__n_estimators": [10, 50, 100],
             "estimator__criterion": ["gini", "entropy", "log_loss"],
             "estimator__max_features": [None, "sqrt", "log2"]
         })
    ]

    learning_curve_params = {
        "X": X_train,
        "y": y_train,
        "train_sizes": np.linspace(0.1, 1.0, 10),
        "cv": 5,
        "scoring": "accuracy",
        "random_state": random_seed
    }

    accuracies = []

    for (model, params) in tqdm(models, desc="Predicting the result "
                                             "(classification)"):
        if hyperparams_tuning:
            model = GridSearchCV(OneVsRestClassifier(model), params, cv=5,
                                 scoring="accuracy", verbose=verbose)
            model.fit(X_train, y_train)
            model = model.best_estimator_
        else:
            model = OneVsRestClassifier(model)
            model.fit(X_train, y_train)

        y_pred_prob = model.predict_proba(X_test)
        y_pred = prob2target(pd.DataFrame(y_pred_prob), t)

        y_pred = y_pred[y_pred != -1].dropna()
        y_test = y_test[y_pred.index]

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append((model.estimator, accuracy))

        save_learning_curve(model, learning_curve_params, n)

    return accuracies


def classification(data, n, hyperparams_tuning, verbose, t):
    targets = pd.read_csv(f"data/ml_targets_result{n}.csv", low_memory=False,
                          header=None)
    accuracies = classification_prediction(data, targets, hyperparams_tuning,
                                           n, verbose, t)

    return sorted(accuracies, key=lambda x: x[1], reverse=True)


def regression_prediction(data, targets, hyperparams_tuning, n, verbose,
                          train_size=0.8):
    train_len = int(data.shape[0] * train_size)

    X_train = data.iloc[:train_len]
    X_test = data.iloc[train_len:]
    y_train = targets.iloc[:train_len]
    y_test = targets.iloc[train_len:]
    y_test.reset_index(drop=True, inplace=True)

    models = [
        (LinearRegression(),
         {}),  # No hyperparameters to tune
        (KNeighborsRegressor(),
         {
             "estimator__n_neighbors": range(1, 11),
             "estimator__weights": ["uniform", "distance"],
             "estimator__p": [1, 2, 3]
         }),
        (SVR(max_iter=1000),
         {
             "estimator__kernel": ["linear", "poly", "rbf", "sigmoid"],
             "estimator__gamma": ["scale", "auto"]
         }),
        (MLPRegressor(random_state=random_seed),
         {
             "estimator__hidden_layer_sizes": [(50,), (100,), (500,), (50, 2),
                                               (100, 2), (500, 2), (50, 3),
                                               (100, 3), (500, 3)],
             "estimator__activation": ["identity", "logistic", "tanh", "relu"],
             "estimator__alpha": [0.0001, 0.001, 0.01, 0.1, 1],
         }),
        (DecisionTreeRegressor(random_state=random_seed),
         {
             "estimator__criterion": ["squared_error", "friedman_mse",
                                      "poisson"],
             "estimator__splitter": ["best", "random"],
             "estimator__max_features": [None, "sqrt", "log2"],
         }),
        (RandomForestRegressor(random_state=random_seed),
         {
             "estimator__n_estimators": [10, 50, 100],
             "estimator__criterion": ["squared_error", "friedman_mse",
                                      "poisson"],
             "estimator__max_features": [None, "sqrt", "log2"]
         })
    ]

    learning_curve_params = {
        "X": X_train,
        "y": y_train,
        "train_sizes": np.linspace(0.1, 1.0, 10),
        "cv": 5,
        "scoring": "neg_mean_absolute_error",
        "random_state": random_seed
    }

    accuracies = []

    for (model, params) in tqdm(models,
                                desc="Predicting the score (regression)"):
        if hyperparams_tuning:
            model = GridSearchCV(MultiOutputRegressor(model), params, cv=5,
                                 scoring="neg_mean_absolute_error",
                                 verbose=verbose)
            model.fit(X_train, y_train)
            model = model.best_estimator_
        else:
            model = MultiOutputRegressor(model)
            model.fit(X_train, y_train)

        y_pred = pd.DataFrame(model.predict(X_test))

        # Round to the nearest integer
        y_pred = y_pred.applymap(lambda x: np.floor(x) if x % 1 < 0.5
                                 else np.ceil(x)).astype(int)
        # Set negative values to 0
        y_pred = y_pred.applymap(lambda x: 0 if x < 0 else x)

        accuracy = my_accuracy_score(y_test, y_pred)
        accuracies.append((model.estimator, accuracy))

        save_learning_curve(model, learning_curve_params, n)

    return accuracies


def regression(data, n, hyperparams_tuning, verbose):
    targets = pd.read_csv(f"data/ml_targets_score{n}.csv", low_memory=False,
                          header=None)
    accuracies = regression_prediction(data, targets, hyperparams_tuning,
                                       n, verbose)

    return sorted(accuracies, key=lambda x: x[1], reverse=True)


def main():
    usage_message = "Usage: python3 predict.py <n> <file name> [--notuning] " \
                    "[--verbose] [--t <threshold>]"
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(usage_message)
        sys.exit(0)
    if len(sys.argv) < 3:
        print(usage_message)
        sys.exit(1)

    n = int(sys.argv[1])
    filename = sys.argv[2]
    check_file_exists(filename)

    start_time = time.time()
    start_time_str = datetime.datetime.now().strftime('%d-%m-%Y %H:%M')
    print(f"Started at: {start_time_str}.")

    hyperparams_tuning = True
    verbose = 0
    t = 0.0

    if len(sys.argv) > 3:
        if "--notuning" in sys.argv:
            hyperparams_tuning = False
            print("Hyperparameters tuning is disabled.")
        if "--verbose" in sys.argv:
            verbose = 3
            print("Verbose mode is enabled.")
        if "--t" in sys.argv:
            t = float(sys.argv[sys.argv.index("--t") + 1])
            print(f"t is set to {t}.")

    data = pd.read_csv(f"data/ml_data{n}.csv", low_memory=False)
    classification_accuracies = classification(data, n, hyperparams_tuning,
                                               verbose, t)
    classification_time = time.time()
    regression_accuracies = regression(data, n, hyperparams_tuning, verbose)

    save_accuracies(classification_accuracies, regression_accuracies, n,
                    filename)
    save_execution_times(start_time, start_time_str, classification_time, n)


if __name__ == "__main__":
    main()
