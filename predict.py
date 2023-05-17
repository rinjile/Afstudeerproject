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
import os
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier  # TODO: OneVsOneClassifier, OutputCodeClassifier
from sklearn.model_selection import GridSearchCV


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


def prediction(data, targets, train_size=0.8):
    train_len = int(data.shape[0] * train_size)

    X_train = data.iloc[:train_len]
    X_test = data.iloc[train_len:]
    y_train = targets.iloc[:train_len]
    y_test = targets.iloc[train_len:]
    y_test.reset_index(drop=True, inplace=True)

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
        model = GridSearchCV(OneVsRestClassifier(model), params, cv=5, scoring="accuracy")

        model.fit(X_train, y_train)
        best_model = model.best_estimator_
        y_pred_prob = best_model.predict_proba(X_test)

        y_pred = prob2target(pd.DataFrame(y_pred_prob))

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append((best_model.estimator, accuracy))

    return accuracies


def main():
    filename = input("File name of the accuracies: ")

    while os.path.exists(f"data/{filename}.txt"):
        filename = input("File name already exists, choose another: ")

    data = pd.read_csv("data/ml_data.csv", low_memory=False)
    targets = pd.read_csv("data/targets.csv", low_memory=False, header=None)
    # data = data.head(50)
    # targets = targets.head(50)

    accuracies = prediction(data, targets)
    accuracies = sorted(accuracies, key=lambda x: x[1], reverse=True)

    with open(f"data/{filename}.txt", "w") as f:
        for (model, accuracy) in accuracies:
            f.write(f"{model}: {accuracy * 100:.2f} %\n")


if __name__ == "__main__":
    main()
