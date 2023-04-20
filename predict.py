"""
Name: Rinji Le
Student ID: 13344552
Course: Afstudeerproject Bachelor Informatica
Date: TODO

Description:
TODO
"""

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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


def predict(data, targets, n=10, train_size=0.8):
    # X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
    # y_test.reset_index(drop=True, inplace=True)

    train_len = int(data.shape[0] * train_size)

    X_train = data.iloc[:train_len]
    X_test = data.iloc[train_len:]
    y_train = targets.iloc[:train_len]
    y_test = targets.iloc[train_len:]
    y_test.reset_index(drop=True, inplace=True)

    models = [
        LogisticRegression(max_iter=10**4),
        SGDClassifier(),
        RidgeClassifier(),
        GaussianNB(),
        BernoulliNB(),
        MultinomialNB(),
        ComplementNB(),
        KNeighborsClassifier(),
        SVC(),
        GaussianProcessClassifier(),
        DecisionTreeClassifier(),
        ExtraTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
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
        accuracy_sum = 0

        for _ in tqdm(range(n), leave=False, desc="Repeating"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy_sum += accuracy_score(y_test, y_pred)

            # model.fit(X_train, y_train)
            # y_pred = pd.DataFrame(model.predict(X_test))
            # correct = (y_test == y_pred).all(axis=1).sum()
            # accuracy_sum += correct / y_test.shape[0]

        accuracies.append((model.__class__.__name__, accuracy_sum / n))

    return accuracies


def main():
    data = pd.read_csv("data/ml_data.csv", low_memory=False)
    targets = pd.read_csv("data/targets.csv", low_memory=False, header=None)

    accuracies = predict(data, targets[0])

    print("Accuracy:")
    for (model, accuracy) in accuracies:
        print(f"- {model}: {accuracy * 100:.2f} %")


if __name__ == "__main__":
    main()
