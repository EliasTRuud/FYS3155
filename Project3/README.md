Project 3
==============================
In this project we explore a dataset containing COVID data (https://www.kaggle.com/datasets/meirnizri/covid19-dataset).
The dataset  contains an enormous number of anonymized patient-related information including pre-conditions. The raw dataset consists of 21 unique features and 1,048,576 unique patients. In the Boolean features, 1 means "yes" and 2 means "no". values as 97 and 99 are missing data.

We've chosen to try to create a target being high risk, which is a combination of 3 of the features: death, intubed and icu. If one of these is present, the pateint is considered high risk. We do a series of function to handle the dataset such as converting age to one hot encoded age groups etc...
Then attempt to use keras and scikit learn models: neural network, logistic regression and decision trees to see if we can find model. Explore some different metrics and the meaning of them.

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
To produce the results LogisticRegression and DecisionTreeClassifier which prints results in terminal run:
```
python main.py
```

For the various different plots once would have to do some adjusting in the code. Examples:
(write something here about what to edit)
```
write something here
```

## Summary of codes

The algorithms are coded into different files:

1. covid_data.csv - Contains all the raw data
2. functions.py - Has some metrics and other small custom functions used in other files.
3. genResults.py - Where we process dataset and make a lot of changes. Such as balancing and creating a target column.
4. main.py - Contains the different models and functions to plot and produce results.
