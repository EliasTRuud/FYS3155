Neural network
==============================
In this project we implement codes which we explore Stochastic Gradient Descent and optimizers for learning rate.
Make a feed forward neural network using basic SGD and apply it to regression(Frankie) and classification problem(sklearn breast cancer).
Search for paramters: lambda and eta. As well as layer vs nodes for classification.


## Requirements

It is necessary to have python3 installed, in particular we tested our code with the following modules installed:
Running python version 3.7.6

Packages:
* `scikit-learn` ver. = 0.22.1
* `numpy` ver. = 1.18.1
* `pandas` ver. = 1.0.1
* `seaborn` ver. = 0.10.0
* `matplotlib` ver. =3.1.3


## Reproduction of results
First clone the repository using

		git clone https://github.com/EliasTRuud/FYS3155

Then navigate to the `Project2Final` folder.
To reproduce the test runs using the Franke function, run which asks user if one wishes to reproduce all figures and then plot.
It will move all plots to Plots/ and to the relevant folder. Filename includes relevant info in it.

		python3 main.py

This should produce all the main plots in the report. A couple additional plots are commented out due to long run time. Simply comment them out to run. Optionally you can run new results with different epochs and batch size etc depending on the task.

## Summary of codes

The algorithms are coded into different files:

1. NeuralNetwork.py -

2. functions.py -

3. genResults.py -

4. GradDescent.py -
