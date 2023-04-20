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
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier


def predict(data, targets):
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
    y_test.reset_index(drop=True, inplace=True)

    models = [
        LogisticRegression(max_iter=10**4),
        GaussianNB(),
        KNeighborsClassifier(),
        SVC(),
        GaussianProcessClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(),
        AdaBoostClassifier()
    ]
    # models = [
    #     # TODO: random state
    #     RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0),
    #     # LogisticRegression(random_state=0, max_iter=10**4),
    #     MultiOutputClassifier(GaussianNB()),
    #     KNeighborsClassifier(3),
    #     # SVC(kernel="linear", C=0.025),
    #     # SVC(gamma=2, C=1),
    #     # GaussianProcessClassifier(1.0 * RBF(1.0)),
    #     DecisionTreeClassifier(max_depth=5),
    #     RandomForestClassifier(),
    #     MultiOutputClassifier(MLPClassifier(alpha=1, max_iter=10**4)),
    #     MultiOutputClassifier(AdaBoostClassifier()),
    #     # QuadraticDiscriminantAnalysis(),
    # ]

    accuracies = []

    for model in tqdm(models, desc="Predicting with the models"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append((model.__class__.__name__, accuracy_score(y_test, y_pred)))

        # model.fit(X_train, y_train)
        # y_pred = pd.DataFrame(model.predict(X_test))

        # correct = (y_test == y_pred).all(axis=1).sum()
        # accuracy = correct / y_test.shape[0]
        # accuracies.append((model.__class__.__name__, accuracy))

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
