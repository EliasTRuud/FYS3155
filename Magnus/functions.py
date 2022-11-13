import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
    sig_x  = sigmoid(x)
    return sig_x*(1 - sig_x)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - tanh(x)**2

def elu(x, alpha=0.01):
    xexp = np.exp(x)
    return np.where(x<0, alpha*(xexp - 1), x)

def elu_deriv(x, alpha=0.01):
    return np.where(x<0, alpha*np.exp(x), 1)

def linear(x):
    return x

def linear_deriv(x):
    return 1

def f(x):
    return 1 + 5*x + 3*x**2

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def scale(X_train, X_test, z_train):
    #Scale data and return it + mean value from target train data.
    scaler = StandardScaler()
    #scaler = MinMaxScaler(feature_range=(-1,1))
    print(X_train.shape)
    X_train = X_train.reshape(1,-1).T
    X_test = X_test.reshape(1,-1).T
    print(X_test.shape)
    scaler.fit(X_train)
    X_train_ = scaler.transform(X_train)
    X_test_ = scaler.transform(X_test)
    z_mean_train = np.mean(z_train)
    X_train_ = X_train.T
    X_test_ = X_test.T
    return X_train_, X_test_, z_mean_train

def MSE(y_data,y_model):
    n = np.size(y_model)
#     print(y_data.shape, y_model.shape)
    return np.sum((y_data-y_model)**2)/n

def softmax(z):
    exp_term = np.exp(z)
    return exp_term/np.sum(exp_term, axis=1, keepdims=True)

def plot_data(x,y,data,title=None):

    # plot results
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)
    
    cbar=fig.colorbar(cax)
    cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
    cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
    cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.1f}\\%$".format( 100*data[j,i])  
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y]


    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)

    ax.set_xlabel('$\\mathrm{Learning\\ rate}$',fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{Regularization\\ parameter}$',fontsize=fontsize)
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    plt.show()

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector