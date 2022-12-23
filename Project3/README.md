Project 3
==============================
In this project we explore a dataset containing COVID data (https://www.kaggle.com/datasets/meirnizri/covid19-dataset).
The dataset


## Requirements

It is necessary to have python3 installed, in particular we tested our code with the following modules installed:
Running python version 3.7.6

Packages:
* `scikit-learn`
* `numpy`
* `pandas`
* `seaborn`
* `matplotlib`
* `tenserflow`
* `tenserflow.keras`


## Reproduction of results
First clone the repository using:
```
git clone https://github.com/EliasTRuud/FYS3155
```
Then navigate to the `Project3` folder.
To produce the results LogisticRegression and DecisionTreeClassifier just run

## Summary of codes

The algorithms are coded into different files:

1. NeuralNetwork.py - Contains the class Layer and NeuralNetwork.

2. functions.py - Contains all the code for the different activation functions. As well as FrankeFunction() and scale() which scales the features and targets.

3. genResults.py - Includes funtions for gridsearch and plotting various figures for regression, classification and  logistic regression.

4. GradDescent.py - Code for various gradient descent with different optimizers. Additionally plotting for the gridsearc.
