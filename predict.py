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
import datetime
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier, \
    LinearRegression, SGDRegressor
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


def save_learning_curve(model, learning_curve_params, ci=95):
    train_sizes, train_scores, validation_scores = learning_curve(
        model, **learning_curve_params)
    train_ci_lower = np.percentile(train_scores, (100 - ci) / 2, axis=1)
    train_ci_upper = np.percentile(train_scores, (100 + ci) / 2, axis=1)
    validation_ci_lower = np.percentile(validation_scores, (100 - ci) / 2,
                                        axis=1)
    validation_ci_upper = np.percentile(validation_scores, (100 + ci) / 2,
                                        axis=1)

    with open(f"results/learning_curve_{model.estimator.__class__.__name__}"
              ".csv", "w") as f:
        f.write("train_size,train_mean,train_ci_lower,train_ci_upper,"
                "validation_mean,validation_ci_lower,validation_ci_upper\n")

        for i in range(train_sizes.shape[0]):
            f.write(f"{train_sizes[i]}," f"{train_scores[i].mean()},"
                    f"{train_ci_lower[i]}," f"{train_ci_upper[i]},"
                    f"{validation_scores[i].mean()},"
                    f"{validation_ci_lower[i]},"
                    f"{validation_ci_upper[i]}\n")


def save_accuracies(classification_accuracies, regression_accuracies,
                    filename):
    with open(f"results/{filename}.csv", "w") as f:
        f.write("type,model,accuracy,hyperparameters\n")

        for (model, accuracy) in classification_accuracies:
            f.write(f"classification,{model.__class__.__name__},"
                    f"{accuracy * 100:.2f},\"{model.get_params()}\"\n")

        for (model, accuracy) in regression_accuracies:
            f.write(f"regression,{model.__class__.__name__},"
                    f"{accuracy * 100:.2f},\"{model.get_params()}\"\n")


def classification_prediction(data, targets, hyperparams_tuning, verbose,
                              train_size=0.8):
    train_len = int(data.shape[0] * train_size)

    X_train = data.iloc[:train_len]
    X_test = data.iloc[train_len:]
    y_train = targets.iloc[:train_len]
    y_test = targets.iloc[train_len:]
    y_test.reset_index(drop=True, inplace=True)

    models = [
        # Max_iter to prevent most convergence warnings
        (LogisticRegression(max_iter=10**4, random_state=random_seed),
         {
             "estimator__C": [0.1, 0.5, 1, 3],
             "estimator__solver": ["lbfgs", "liblinear", "newton-cg",
                                   "newton-cholesky", "sag", "saga"]
         }),
        (GaussianNB(),
         {}),  # No hyperparameters to tune
        (BernoulliNB(),  # Boolean features (TODO: deze doen?)
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
             # TODO: experiment met 1 tm 10
             "estimator__n_neighbors": [1, 2, 4, 5, 7, 8, 10],
             "estimator__weights": ["uniform", "distance"],
             # "estimator__algorithm": ["ball_tree", "kd_tree", "brute"],
             # "estimator__leaf_size": [10, 20, 30, 40, 50],
             "estimator__p": [1, 2, 3]
         }),
        # TODO: niet doen?
        (SGDClassifier(loss="log_loss", random_state=random_seed),
         {
             # "estimator__penalty": ["l2", "l1", "elasticnet"],
             "estimator__alpha": [0.0001, 0.001, 0.01, 0.1, 1],
             "estimator__learning_rate": ["constant", "optimal", "invscaling",
                                          "adaptive"],
             "estimator__power_t": [0.1, 0.2, 0.3, 0.4, 0.5]
         }),
        # TODO: experiment zonder max_iter
        (SVC(probability=True, max_iter=1000, random_state=random_seed),
         {
             # "estimator__C": [0.1, 1, 10, 100],
             "estimator__kernel": ["linear", "poly", "rbf", "sigmoid"],
             # "estimator__degree": [1, 2, 3, 4, 5],
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

        # GaussianProcessClassifier(),
        # RidgeClassifier(),
        # ComplementNB(),
        # ExtraTreeClassifier(),
        # AdaBoostClassifier(),
    ]

    learning_curve_params = {
        "X": X_train,
        "y": y_train,
        "train_sizes": np.linspace(0.1, 1.0, 10),
        "cv": 5,
        "scoring": "accuracy",
        "random_state": random_seed
        # "n_jobs": 4,
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
        y_pred = prob2target(pd.DataFrame(y_pred_prob))

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append((model.estimator, accuracy))

        save_learning_curve(model, learning_curve_params)

    return accuracies


def classification(data, hyperparams_tuning, verbose):
    targets = pd.read_csv("data/ml_targets_result.csv", low_memory=False,
                          header=None)
    # data = data.head(50)
    # targets = targets.head(50)

    accuracies = classification_prediction(data, targets, hyperparams_tuning,
                                           verbose)
    return sorted(accuracies, key=lambda x: x[1], reverse=True)


def regression_prediction(data, targets, hyperparams_tuning, verbose,
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
             # TODO: experiment met 1 tm 10
             "estimator__n_neighbors": [1, 2, 4, 5, 7, 8, 10],
             "estimator__weights": ["uniform", "distance"],
             "estimator__p": [1, 2, 3]
         }),
        (SGDRegressor(random_state=random_seed),  # TODO: niet doen?
         {
             "estimator__alpha": [0.0001, 0.001, 0.01, 0.1, 1],
             "estimator__learning_rate": ["constant", "optimal", "invscaling",
                                          "adaptive"],
             "estimator__power_t": [0.1, 0.2, 0.3, 0.4, 0.5]
         }),
        (SVR(max_iter=1000),  # TODO: experiment zonder max_iter
         {
             # "estimator__C": [0.1, 1, 10, 100],
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
                                      "absolute_error", "poisson"],
             "estimator__splitter": ["best", "random"],
             "estimator__max_features": [None, "sqrt", "log2"],
         }),
        (RandomForestRegressor(random_state=random_seed),
         {
             "estimator__n_estimators": [10, 50, 100],
             "estimator__criterion": ["squared_error", "friedman_mse",
                                      "absolute_error", "poisson"],
             "estimator__max_features": [None, "sqrt", "log2", 1]
         })

        # (GradientBoostingRegressor(random_state=random_seed),
        #  {})
        # BayesianRidge
    ]

    learning_curve_params = {
        "X": X_train,
        "y": y_train,
        "train_sizes": np.linspace(0.1, 1.0, 10),
        "cv": 5,
        "scoring": "neg_mean_absolute_error",  # TODO: andere?
        "random_state": random_seed
        # "n_jobs": 4,
    }

    accuracies = []

    for (model, params) in tqdm(models,
                                desc="Predicting the score (regression)"):
        if hyperparams_tuning:
            # TODO: andere scoring?
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
        # (TODO: experiment met afronden naar 0)

        accuracy = my_accuracy_score(y_test, y_pred)
        accuracies.append((model.estimator, accuracy))

        save_learning_curve(model, learning_curve_params)

    return accuracies


def regression(data, hyperparams_tuning, verbose):
    targets = pd.read_csv("data/ml_targets_score.csv", low_memory=False,
                          header=None)
    # data = data.head(500)
    # targets = targets.head(500)

    accuracies = regression_prediction(data, targets, hyperparams_tuning,
                                       verbose)
    return sorted(accuracies, key=lambda x: x[1], reverse=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <file name> [--notuning] [--verbose]")
        sys.exit(1)

    filename = sys.argv[1]
    check_file_exists(filename)

    print(f"Started at: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M')}.")

    hyperparams_tuning = True
    verbose = 0

    if len(sys.argv) > 2:
        if "--notuning" in sys.argv:
            hyperparams_tuning = False
            print("Hyperparameters tuning is disabled.")
        if "--verbose" in sys.argv:
            verbose = 3
            print("Verbose mode is enabled.")

    data = pd.read_csv("data/ml_data.csv", low_memory=False)
    classification_accuracies = classification(data, hyperparams_tuning,
                                               verbose)
    regression_accuracies = regression(data, hyperparams_tuning, verbose)
    # classification_accuracies = []
    # regression_accuracies = []
    save_accuracies(classification_accuracies, regression_accuracies, filename)


if __name__ == "__main__":
    main()
