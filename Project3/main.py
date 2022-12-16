from genResults import get_df
# from functions import R_squared
from tensorflow_addons.metrics import RSquare
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from functions import get_f1, matthews_correlation
# from tfa.metrics import MatthewsCorrelationCoefficient 
import tensorflow as tf
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical

seed = 12345
np.random.seed(seed)

def NN_model(input_size, metrics, grid_search, n_layers=None, n_neuron=None, eta=None, lamda=None, activation_func="relu"):
    """
    Taken from lecture notes. Creates a NN model using keras.
    """
    for parameter, parameter_name in grid_search:
        if parameter_name == "eta":
            eta = parameter
        elif parameter_name == "lamda":
            lamda = parameter
        elif parameter_name == "n_layers":
            n_layers = parameter
        elif parameter_name == "n_neurons":
            n_neuron = parameter
    model = Sequential()
    for i in range(n_layers):
        if i==0:
            model.add(Dense(n_neuron, activation=activation_func, kernel_regularizer=regularizers.l2(lamda), input_dim=input_size))
        else:
            model.add(Dense(n_neuron, activation=activation_func, kernel_regularizer=regularizers.l2(lamda)))
    model.add(Dense(2, activation="softmax"))
    sgd = optimizers.SGD(learning_rate=eta)
    # "adam"
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=metrics)
    return model

def keras_NN(X_train, X_test, y_train, y_test, metrics=["accuracy"], epochs = 10, batch_size = 100, grid_search=None):
    """
    Taken from lecture notes. Trains the model and collects metrics. metrics must be of type list.
    The grid_search feature lets you seach for the best combination of two different parameters,
    it does not work with more than two. The feature takes in four parameters, which are assumed to
    be eta, n_neurons, n_layers and lamda, other variables are not supported.
    """
    if not(isinstance(metrics, list)): 
        raise TypeError
    
    #define tunable parameters
    if not(isinstance(grid_search, list)):
        parameter1 = np.logspace(-3, -1, 3) #eta
        parameter2 = np.logspace(0, 3, 4, dtype=int) #n_neurons
        parameter3 = 2 #n_layers
        parameter4 = 0.01 #lambda
        parameters = [parameter1, parameter2, parameter3, parameter4]
        parameters_names = ["eta", "n_neurons", "n_layers", "lamda"]
    else:
        parameters = [0, 0, 0, 0]
        parameters_names = [0, 0, 0, 0]
        i = 0
        j = 2
        for parameter, parameter_name in grid_search:
            if isinstance(parameter, np.ndarray):
                parameters[i] = parameter
                parameters_names[i] = parameter_name
                i += 1
            else:
                parameters[j] = parameter
                parameters_names[j] = parameter_name
                j += 1
        parameter1, parameter2, parameter3, parameter4 = parameters
    

    train_accuracy = np.zeros((len(parameter2), len(parameter1)))
    test_accuracy = np.zeros((len(parameter2), len(parameter1)))

    for i in range(len(parameter2)):
        for j in range(len(parameter1)):
            print("combination: ", j+i*len(parameter1)+1, "out of ", len(parameter1)*len(parameter2),
                  f"\n{parameters_names[0]}={parameter1[j]}, {parameters_names[1]}={parameter2[i]}")
            DNN_model = NN_model(X_train.shape[1], metrics=metrics, 
                                grid_search=[[parameter1[j], parameters_names[0]], [parameter2[i], parameters_names[1]],
                                             [parameter3, parameters_names[2]], [parameter4, parameters_names[3]]]
                                )
            DNN_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.15, verbose=1)
            train_accuracy[i, j] = DNN_model.evaluate(X_train, y_train)[1]
            test_accuracy[i, j] = DNN_model.evaluate(X_test, y_test)[1]

    train_accuracy_df = pd.DataFrame(train_accuracy, columns=parameter1, index=parameter2)
    test_accuracy_df = pd.DataFrame(test_accuracy, columns=parameter1, index=parameter2)
    return train_accuracy_df, test_accuracy_df

def plot_data(data, labels, title=None):
    """
    Plots the data so that we can determine their
    """
    plt.rc('axes', titlesize=16)
    plt.subplots_adjust(hspace=0.1)
    fig, ax= plt.subplots(figsize=(8, 8), sharey=True, tight_layout=True)
    ax.set_title(title)
    ax = sns.heatmap(data, ax=ax, cbar=True, annot=True, annot_kws={"fontsize":11}, fmt=".3%")
    ax.set(xlabel=labels[0], ylabel=labels[1])
    fig.subplots_adjust(wspace=0.001)
    plt.show()

if __name__ == "__main__":
    df = get_df("covid_data.csv", n=10000)
    # df = df.sample(n=10000, random_state=seed)
    print(df["HIGH_RISK"].value_counts()[1], df["HIGH_RISK"].value_counts()[0])
    target = df["HIGH_RISK"]
    inputs = df.loc[:, df.columns != "HIGH_RISK"]

    X = inputs
    y = target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print("Total percentage not high risk ",df.loc[df["HIGH_RISK"] == 1].shape[0]/df.shape[0]*100, "%")

    epochs = 10
    batch_size = 100
    # metrics = [matthews_correlation] #, RSquare]
    metrics = ["accuracy"]

    eta = np.logspace(-3, 2, 6) #eta
    # eta = 0.1
    # n_neurons = np.logspace(1, 2, 2, dtype=int) #n_neurons
    n_neurons = 10
    # n_layers = np.linspace(2, 5, 3, dtype=int) #n_layers
    n_layers = 3
    # lamda = 0.01 #lambda
    lamda = np.logspace(-3, 2, 6)

    grid_search = [[eta, "eta"], [n_neurons, "n_neurons"], [n_layers, "n_layers"], [lamda, "lamda"]]    

    train_accuracy, test_accuracy = keras_NN(X_train, X_test, y_train, y_test, metrics=metrics, epochs=epochs, 
                                             batch_size=batch_size, grid_search=grid_search)

    plot_data(train_accuracy, [feature_name for feature, feature_name in grid_search if isinstance(feature, np.ndarray)], "Training")
    plot_data(test_accuracy, [feature_name for feature, feature_name in grid_search if isinstance(feature, np.ndarray)], "Testing")