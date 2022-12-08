from genResults import get_df
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

def NN_model(input_size, n_layers, n_neuron, eta, lamda, metrics, activation_func="relu"):
    """
    Taken from lecture notes. Creates a NN model using keras.
    """
    model = Sequential()
    for i in range(n_layers):
        if i==0:
            model.add(Dense(n_neuron, activation=activation_func, kernel_regularizer=regularizers.l2(lamda), input_dim=input_size))
        else:
            model.add(Dense(n_neuron, activation=activation_func, kernel_regularizer=regularizers.l2(lamda)))
    model.add(Dense(2, activation="softmax"))
    sgd = optimizers.SGD(learning_rate=eta)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=metrics)
    return model

def keras_NN(X_train, X_test, y_train, y_test, metrics=["accuracy"], epochs = 10, batch_size = 100):
    """
    Taken from lecture notes. Trains the model and collects metrics. metrics must be of type list.
    """
    if not(isinstance(metrics, list)): 
        raise TypeError
    #define tunable parameters
    eta = np.logspace(-3, -1, 3)
    lamda = 0.01
    n_layers = 2
    n_neuron = np.logspace(0, 3, 4, dtype=int)

    

    train_accuracy = np.zeros((len(n_neuron), len(eta)))
    test_accuracy = np.zeros((len(n_neuron), len(eta)))

    for i in range(len(n_neuron)):
        for j in range(len(eta)):
            print(j+i*len(eta))
            DNN_model = NN_model(X_train.shape[1], n_layers, n_neuron[i], eta[j], lamda, metrics=metrics)
            DNN_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
            train_accuracy[i, j] = DNN_model.evaluate(X_train, y_train)[1]
            test_accuracy[i, j] = DNN_model.evaluate(X_test, y_test)[1]

    train_accuracy_df = pd.DataFrame(train_accuracy, columns=eta, index=n_neuron)
    test_accuracy_df = pd.DataFrame(test_accuracy, columns=eta, index=n_neuron)
    return train_accuracy_df, test_accuracy_df

def plot_data(data,title=None):
    plt.rc('axes', titlesize=16)
    plt.subplots_adjust(hspace=0.1)
    fig, ax= plt.subplots(figsize=(8, 8), sharey=True, tight_layout=True)
    ax.set_title(title)
    ax = sns.heatmap(data, ax=ax, cbar=True, annot=True, annot_kws={"fontsize":11}, fmt=".3%")
    ax.set(xlabel="eta", ylabel="n_neuron")
    fig.subplots_adjust(wspace=0.001)
    plt.show()

# #from proj 2
# import NeuralNetwork
# from fromProj2.genResults import plotEtaLambda
# plotEtaLambda(10, data=df)

if __name__ == "__main__":
    df = get_df("covid_data.csv")
    df = df.sample(n=10000, random_state=seed)

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

    train_accuracy, test_accuracy = keras_NN(X_train, X_test, y_train, y_test, metrics=[matthews_correlation], epochs=epochs, batch_size=batch_size)
    plot_data(train_accuracy, "Training")
    plot_data(test_accuracy, "Testing")