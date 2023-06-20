# Data Goals: Predictive Football with Advanced Statistics

## Description
In this project, football matches from the top 5 European leagues 
(England, Spain, Germany, Italy, and France) are predicted using
different machine learning algorithms. The seasons 2015/2016 to
and including 2021/2022 are used for training and testing. The first
80% of all the matches are used for training and the last 20% for
testing. The data are taken from is from [API-Football](https://www.api-football.com).

The used algorithms are: 
- Naive Bayes
- Linear Regression
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Neural Network
- Decision Tree
- Random Forest

The features set is based on the last *n* matches of both teams and
the last *n* matches between the two teams. Muitiple *n* are used
to find the best value for *n*. The experiments are done with *n*
ranging from 1 to and including 10. The features are the match score 
and statistics like shots on target, corners, cards, 
possession, etc.

A grid search is used to find the best hyperparameters for each
algorithm. In this grid search, the 5-fold cross-validation is applied
to the training data. Furthermore, the learning curves of the
algorithms are plotted to see how the performance of the algorithms
changes with the training set size.

## Requirements
The required Python packages are listed in the `requirements.txt` file.
To install them, run the following command:
```
pip install -r requirements.txt
```

## Files
The project contains 4 main files:
- `api-football.py`: This file collects the data from [API-Football](https://www.api-football.com).
  The data are saved in the `data/` folder. (Note: Since the API key is
  not provided, this file cannot be used. The required data are already
    provided in `data/`.)
- `ml_data.py`: This file creates the feature set and the target set.
  The data are saved in the `data/` folder.
- `predict.py`: This file trains the algorithms and predicts the matches. 
  The results are saved in the `results/` folder. (Note: The results of the 
  experiments are already provided in `results/`.)
- `plot.py`: This file plots the learning curves and the bar charts of 
  the accuracy of the algorithms. The plots are saved in the `plots/` folder.

## Usage
All the files are provided with a help message for the usage.
To see the help message, add the argument `--help` to the command.
In this given message, `< >` means that the argument needs to be replaced
and `[ ]` means that the argument is optional.

Some examples:
- Create the dataset for the last 3 matches:
```
python ml_data.py 3
```
- Predict the matches with the dataset for the last 5 matches and
  save the results as `accuracies.csv` without tuning the hyperparameters:
```
python predict.py 5 accuracies --notuning
```
- Plot the bar charts of the results, which are saved in `results/n10/accuracies10.csv`,
  from the dataset for the last 10 matches:
```
python plot.py 10 results/n10/accuracies10.csv
```
- Plot the accuracies of the algorithms for the values of *n* from 1 to and including 10:
```
  python plot.py --n
```
