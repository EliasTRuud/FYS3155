from genResults import get_df
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical


df = get_df("covid_data.csv")

target = df["HIGH_RISK"]
inputs = df.loc[:, df.columns != "HIGH_RISK"]

X = inputs
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#define tunable parameters
eta = np.logspace(-3, -1, 3)
lamda = 0.01
n_layers = 2
n_neuron = np.logspace(0, 3, 4, dtype=int)
epochs = 10
batch_size = 100

def NN_model(input_size, n_layers, n_neuron, eta, lamda, activation_func="relu"):
    model = Sequential()
    for i in range(n_layers):
        if i==0:
            model.add(Dense(n_neuron, activation=activation_func, kernel_regularizer=regularizers.l2(lamda), input_dim=input_size))
        else:
            model.add(Dense(n_neuron, activation=activation_func, kernel_regularizer=regularizers.l2(lamda)))
    model.add(Dense(2, activation="softmax"))
    sgd = optimizers.SGD(learning_rate=eta)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model

train_accuracy = np.zeros((len(n_neuron), len(eta)))
test_accuracy = np.zeros((len(n_neuron), len(eta)))

for i in range(len(n_neuron)):
    for j in range(len(eta)):
        print(X_train.shape)
        DNN_model = NN_model(X_train.shape[1], n_layers, n_neuron[i], eta[j], lamda)
        DNN_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        train_accuracy[i, j] = DNN_model.evaluate(X_train, y_train)[1]
        train_accuracy[i, j] = DNN_model.evaluate(X_test, y_test)[1]

def plot_data(x,y,data,title=None):

    # plot results
    fontsize = 16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)
    
    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
    cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
    cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.1f}\\%$".format( 100*data[j,i])  
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x = [str(i) for i in x]
    y = [str(i) for i in y]


    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)

    ax.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{hidden\\ neurons}$',fontsize=fontsize)
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    plot_data(eta,n_neuron, train_accuracy, 'training')
    plot_data(eta,n_neuron, test_accuracy, 'testing')