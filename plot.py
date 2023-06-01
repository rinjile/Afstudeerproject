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

classifiers = [
    "LogisticRegression",
    "GaussianNB",
    "BernoulliNB",
    "MultinomialNB",
    "KNeighborsClassifier",
    "SGDClassifier",
    "SVC",
    "MLPClassifier",
    "DecisionTreeClassifier",
    "RandomForestClassifier"
]

# TODO: regression
model_names = {
    "LogisticRegression": "Logistische Regressie",
    "GaussianNB": "Gaussian Naive Bayes",
    "BernoulliNB": "Bernoulli Naive Bayes",
    "MultinomialNB": "Multi-nomiaal Naive Bayes",
    "KNeighborsClassifier": "K-nearest Neighbors",
    "SGDClassifier": "SGDClassifier",  # TODO
    "SVC": "Support Vector Machine",
    "MLPClassifier": "Neuraal Netwerk",
    "DecisionTreeClassifier": "Decision Tree",
    "RandomForestClassifier": "Random Forest"
}


def plot_bars(data):
    data = data.sort_values(by="accuracy", ascending=True)
    models = data["model"]
    accuracies = data["accuracy"]

    bars = plt.barh([model_names[model] for model in models], accuracies, color="royalblue")
    plt.bar_label(bars, label_type="center", fmt="%.2f")
    plt.title("Nauwkeurigheid van verschillende modellen")
    plt.xlabel("Nauwkeurigheid (%)")
    plt.tight_layout()
    plt.savefig("plots/accuracies.png", dpi=1000)


def plot_learning_curve(x, y, ci_lower, ci_upper, model):
    plt.plot(x, y[0], label="Train-score", marker="o", color="royalblue")
    plt.fill_between(x, ci_lower[0], ci_upper[0], label="95% CI (train)", alpha=0.2, color="royalblue")
    plt.plot(x, y[1], label="Validatie-score", marker="o", color="limegreen")
    plt.fill_between(x, ci_lower[1], ci_upper[1], label="95% CI (validatie)", alpha=0.2, color="limegreen")

    plt.title(f"{model_names[model]}: leercurve met 5-voudige kruisvalidatie")
    plt.xlabel("Trainset grootte")

    if model in classifiers:
        plt.ylabel("Nauwkeurigheid")
    else:
        plt.ylabel("Gemiddelde absolute fout (negatie)")

    plt.legend(loc="best")
    plt.savefig(f"plots/learning_curve_{model}.png", dpi=1000)
    plt.clf()  # Clear the figure


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot.py --bar <file>")
        print("       python plot.py --lc [file]")
        return

    if sys.argv[1] == "--bar":
        if len(sys.argv) < 3:
            print("Usage: python plot.py --bar <file>")
            return
        else:
            data = pd.read_csv(sys.argv[2])
            plot_bars(data)
    elif sys.argv[1] == "--lc":
        if len(sys.argv) > 2:
            file = sys.argv[2]
            files = [file.split("/")[1]]
        else:
            files = os.listdir("results")
            files = [file for file in files if file[:14] == "learning_curve"]

        for file in tqdm(files, desc="Plotting learning curves"):
            model = file.split("_")[-1].split(".")[0]
            data = pd.read_csv(f"results/{file}")

            plot_learning_curve(data["train_size"], [data["train_mean"], data["validation_mean"]],
                                [data["train_ci_lower"], data["validation_ci_lower"]],
                                [data["train_ci_upper"], data["validation_ci_upper"]], model)


if __name__ == "__main__":
    main()
