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


def prediction(data, targets, n=10, train_size=0.8):
    train_len = int(data.shape[0] * train_size)

    X_train = data.iloc[:train_len]
    X_test = data.iloc[train_len:]
    y_train = targets.iloc[:train_len]
    y_test = targets.iloc[train_len:]
    y_test.reset_index(drop=True, inplace=True)

    models = [
        LogisticRegression(max_iter=10**4),
        # SGDClassifier(),
        # RidgeClassifier(),
        GaussianNB(),
        BernoulliNB(),
        # MultiOutputClassifier(MultinomialNB()),
        # MultiOutputClassifier(ComplementNB()),
        KNeighborsClassifier(),
        # SVC(),
        # MultiOutputClassifier(GaussianProcessClassifier()),
        DecisionTreeClassifier(),
        # ExtraTreeClassifier(),
        RandomForestClassifier(),
        # MultiOutputClassifier(AdaBoostClassifier()),
        MLPClassifier(max_iter=10**4)
    ]
    # models = [
    #     RandomForestClassifier(),
    #     MultiOutputClassifier(GaussianNB()),
    #     KNeighborsClassifier(),
    #     DecisionTreeClassifier(),
    #     RandomForestClassifier(),
    #     MultiOutputClassifier(MLPClassifier(max_iter=10**4)),
    #     MultiOutputClassifier(AdaBoostClassifier()),
    # ]

    accuracies = []

    for model in tqdm(models, desc="Predicting with the models"):
        model_name = model.__class__.__name__
        model = OneVsRestClassifier(model)
        accuracy_sum = 0

        for _ in tqdm(range(n), leave=False, desc="Repeating"):
            model.fit(X_train, y_train)
            y_pred_prob = model.predict_proba(X_test)
            y_pred = prob2target(pd.DataFrame(y_pred_prob))
            accuracy_sum += accuracy_score(y_test, y_pred)

        accuracies.append((model_name, accuracy_sum / n))

    return accuracies


def main():
    data = pd.read_csv("data/ml_data.csv", low_memory=False)
    targets = pd.read_csv("data/targets.csv", low_memory=False, header=None)

    accuracies = prediction(data, targets)
    accuracies = sorted(accuracies, key=lambda x: x[1], reverse=True)

    print("Accuracy:")
    for (model, accuracy) in accuracies:
        print(f"- {model}: {accuracy * 100:.2f} %")


if __name__ == "__main__":
    main()
