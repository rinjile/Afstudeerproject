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
# The model names are used for the plots (in Dutch)
model_names = {
    "LogisticRegression": "Logistische Regressie",
    "GaussianNB": "Gaussian Naive Bayes",
    "BernoulliNB": "Bernoulli Naive Bayes",
    "MultinomialNB": "Multi-nomiaal Naive Bayes",
    "KNeighborsClassifier": "K-nearest Neighbors",
    "SGDClassifier": "SGDClassifier",  # TODO: verwijderen?
    "SVC": "Support Vector Machine",
    "MLPClassifier": "Neuraal Netwerk",
    "DecisionTreeClassifier": "Decision Tree",
    "RandomForestClassifier": "Random Forest",

    "LinearRegression": "Lineaire Regressie",
    "SVR": "Support Vector Machine",
    "SGDRegressor": "SGDRegressor",  # TODO
    "GradientBoostingRegressor": "GradientBoostingRegressor"  # TODO
}


def plot_bars(data, model_type="all"):
    if model_type == "classification":
        data = data[data["type"] == model_type]
        title = "Nauwkeurigheid van verschillende classificators"
    elif model_type == "regression":
        data = data[data["type"] == model_type]
        title = "Nauwkeurigheid van verschillende regressors"
    else:
        title = "Nauwkeurigheid van verschillende modellen"

    data = data.sort_values(by="accuracy", ascending=True)
    colors = ["royalblue" if row["type"] == "classification" else "limegreen" for (_, row) in data.iterrows()]

    bars = plt.barh([model_names[model] for model in data["model"]], data["accuracy"], color=colors, edgecolor="black", alpha=0.8)
    plt.bar_label(bars, label_type="center", fmt="%.2f")
    plt.title(title)
    plt.xlabel("Nauwkeurigheid (%)")

    # Add legend if both classifiers and regressors are present in the data
    if model_type == "all" and len(data["type"].unique()) == 2:
        labels = ["Classificator", "Regressor"]
        colors = ["royalblue", "limegreen"]
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.8) for i in range(len(labels))]
        plt.legend(handles, labels, loc="best")

    plt.tight_layout()
    plt.savefig(f"plots/accuracies_{model_type}.png", dpi=1000)
    plt.clf()  # Clear the figure


def plot_learning_curve(data, model):
    # Multiply the data by 100 to get the accuracy in percentages
    if model in classifiers:
        data = data.apply(lambda x: x * 100 if x.name != "train_size" else x)

    plt.plot(data["train_size"], data["train_mean"], label="Train-score", marker="o", color="royalblue")
    plt.fill_between(data["train_size"], data["train_ci_lower"], data["train_ci_upper"], label="95% CI (train)",
                     alpha=0.2, color="royalblue")
    plt.plot(data["train_size"], data["validation_mean"], label="Validatie-score", marker="o", color="limegreen")
    plt.fill_between(data["train_size"], data["validation_ci_lower"], data["validation_ci_upper"], label="95% CI (validatie)",
                     alpha=0.2, color="limegreen")

    plt.title(f"{model_names[model]}: leercurve met 5-voudige kruisvalidatie")
    plt.xlabel("Trainset grootte")

    if model in classifiers:
        plt.ylabel("Nauwkeurigheid (%)")
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
            plot_bars(data, model_type="classification")
            plot_bars(data, model_type="regression")
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
            plot_learning_curve(data, model)


if __name__ == "__main__":
    main()
