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
import os


def plot(x, y, ci_lower, ci_upper, model):
    for i in range(len(y)):
        label = "Training score" if i == 0 else "Test score"
        plt.plot(x, y[i], label=label, marker="o")
        plt.fill_between(x, ci_lower[i], ci_upper[i], label="95% confidence interval", alpha=0.2)

    plt.title(f"Learning curve for {model} with 5-fold cross-validation")
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.savefig(f"plots/learning_curve_{model}.png", dpi=1000)
    plt.clf()  # Clear the figure


def main():
    files = os.listdir("results")

    for file in files:
        model = file.split("_")[-1].split(".")[0]
        data = pd.read_csv(f"results/{file}")

        plot(data["train_size"], [data["train_mean"], data["test_mean"]],
             [data["train_ci_lower"], data["test_ci_lower"]],
             [data["train_ci_upper"], data["test_ci_upper"]], model)


if __name__ == "__main__":
    main()
