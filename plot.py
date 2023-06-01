"""
Name: Rinji Le
Student ID: 13344552
Course: Afstudeerproject Bachelor Informatica
Date: TODO

Description:
TODO
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

# TODO: regression
model_names = {
    "LogisticRegression": "Logistische Regressie",
    "GaussianNB": "Gaussian Naive Bayes",
    "BernoulliNB": "Bernoulli Naive Bayes",
    "MultinomialNB": "Multi-nomiaal Naive Bayes",
    "KNeighborsClassifier": "K-Nearest Neighbors",
    "SGDClassifier": "...",  # TODO
    "SVC": "Support Vector Machine",
    "MLPClassifier": "Neuraal Netwerk",
    "DecisionTreeClassifier": "Decision Tree",
    "RandomForestClassifier": "Random Forest",
}

def plot(x, y, ci_lower, ci_upper, model):
    plt.plot(x, y[0], label="Train-score", marker="o")
    plt.fill_between(x, ci_lower[0], ci_upper[0], label="95% CI (train)", alpha=0.2)
    plt.plot(x, y[1], label="Validatie-score", marker="o")
    plt.fill_between(x, ci_lower[1], ci_upper[1], label="95% CI (validatie)", alpha=0.2)

    # TODO: model naam aanpassen
    plt.title(f"{model_names[model]}: leercurve met 5-voudige kruisvalidatie")
    plt.xlabel("Trainset grootte")
    plt.ylabel("Nauwkeurigheid")  # TODO: bij regression anders
    plt.legend(loc="best")
    plt.savefig(f"plots/learning_curve_{model}.png", dpi=1000)
    plt.clf()  # Clear the figure


def main():
    if len(sys.argv) > 1:
        file = sys.argv[1]
        files = [file.split("/")[1]]
    else:
        files = os.listdir("results")
        files = [file for file in files if file[:14] == "learning_curve"]

    for file in tqdm(files, desc="Plotting learning curves"):
        model = file.split("_")[-1].split(".")[0]
        data = pd.read_csv(f"results/{file}")

        plot(data["train_size"], [data["train_mean"], data["validation_mean"]],
             [data["train_ci_lower"], data["validation_ci_lower"]],
             [data["train_ci_upper"], data["validation_ci_upper"]], model)


if __name__ == "__main__":
    main()
