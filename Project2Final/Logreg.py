import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import load_breast_cancer
from NeuralNetwork import Layer

seed = 32455
np.random.seed(seed)

class LinRegClass:
    def __init__(self, X_data, Y_data, sigma, sigma_d, epochs=100, batch_size=100, eta=0.1, lmbd=0):
        if len(X_data.shape) == 2:
            self.X_data_full = X_data
        else:
            self.X_data_full = X_data.reshape(-1, 1)
        if len(Y_data.shape) == 2:
            self.Y_data_full = Y_data
        else:
            self.Y_data_full = Y_data.reshape(-1, 1)

        np.random.seed(seed)
        self.n_inputs = self.X_data_full.shape[0]
        self.n_features = self.X_data_full.shape[1]
        self.n_outputs = self.Y_data_full.shape[1]

        self.layers = [Layer(self.n_features, self.n_outputs, sigma, sigma_d)]
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta #learning rate
        self.lmbd = lmbd
        self.accuracyTest = []
        self.accTr = []

        self.lossTest = []
        self.lossTrain = []

    def feedForward(self):
        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias

        z = np.matmul(self.X_data, weights) + bias
        layer1.get_z = z
        z_ = layer1.sigma(z)
        layer1.get_a = np.piecewise(z_, [z_ < 0.5, z_ >= 0.5], [0, 1])
        a = [layer1.get_a]
        self.output = a[-1]

    def feedForwardOut(self, X):
        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias
        z = np.matmul(X, weights) + bias
        layer1.get_z = z
        z_ = layer1.sigma(z)
        layer1.get_a = np.piecewise(z_, [z_ < 0.5, z_ >= 0.5], [0, 1]) # 0 if activation is lower than .5, 1 if higher
        a = [layer1.get_a]
        return a

    def backProp(self):
        #Assumes just a input and output layer.
        Y_data = self.Y_data
        error_output = (self.output - Y_data) #Dervation of OLS
        error = [error_output]
        outLayer = self.layers[-1]

        ah = self.X_data
        w_grad = np.matmul(ah.T, error[-1])

        bias_grad_output = np.sum(error_output, axis=0)
        bias_grad = bias_grad_output

        weights_ = outLayer.get_weights
        bias_ = outLayer.get_bias

        outLayer.get_weights = weights_ - self.eta*(w_grad + self.lmbd*weights_*2)
        outLayer.get_bias = bias_ - self.eta*(bias_grad + self.lmbd*bias_*2)



    def train(self, X_test = None, Y_test = None, calcAccuracy = False):
        data_indices = np.arange(self.n_inputs)
        #Loop over epochs(i), with minibatches = batch_size, train network with backProp
        k = 0
        for i in range(self.epochs):
            if i == self.epochs-1:
                print(f"Epochs {self.epochs}/{self.epochs}")
                print("Done.\n")
            if i>=k:
                print(f"Epochs {i}/{self.epochs}")
                k += int(self.epochs/5)

            for j in range(self.iterations):
                chosen_data_points = np.random.choice(data_indices, size=self.batch_size, replace=False)
                #chosen_data_points = np.random.choice(data_indices, size=self.X_data_full.shape[0], replace=False)

                self.X_data = self.X_data_full[chosen_data_points]
                self.Y_data = self.Y_data_full[chosen_data_points]

                self.feedForward()
                self.backProp()
            if calcAccuracy: #Per epoch calc MSE.
                predTrain = self.predict(self.X_data_full)
                predTrain = np.ravel(predTrain[0])
                y_data = np.ravel(self.Y_data_full)
                accTr = accuracy_score(y_data, predTrain)
                self.accTr.append(accTr)

                pred = self.predict(X_test)
                pred = np.ravel(pred[0])
                accT = accuracy_score(Y_test, pred)
                self.accuracyTest.append(accT)

                y_tilde = self.output
                y = Y_test # y_train
                loss = -np.mean(y*np.log(y_tilde+1e-9) + (1-y)*np.log(1-y_tilde+1e-9))
                self.lossTest.append(loss)
                """
                y_tilde = self.output
                y = Y_test
                loss = -np.mean(y*np.log(y_tilde) + (1-y)*np.log(1-y_tilde))
                """
        #plt.plot(np.arange(self.epochs), self.accuracyTest)
        print(self.accuracyTest[-1])
        #plt.show()

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        output = self.feedForwardOut(X)
        return output

    def get_ACCtest(self):
        arr_outEpochs = np.array(self.accuracyTest)
        return arr_outEpochs

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
    sig_x  = sigmoid(x)
    return sig_x*(1 - sig_x)

def relu(x):
    return (np.maximum(0, x))

def relu_deriv(x):
    x_ = (x > 0) * 1
    return x_

def leaky_relu(x):
    if x>0:
        return x
    else:
        return 0.01*x

def tanh_function(x):
    z = (2/(1 + np.exp(-2*x))) -1
    return z

def tanh_deriv(x):
    return 1 - (tanh_function(x))**2

def softmax_function(x):
    z = np.exp(x)
    z_ = z/z.sum()
    return z_

def linear(x):
    return x

def linear_deriv(x):
    return 1

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def scale(X_train, X_test, Y_train, Y_test):
	#Scale data and return it + mean value from target train data.
    if len(X_train.data.shape) <= 1:
        X_train_ = X_train.reshape(-1,1)
        X_test_ = X_test.reshape(-1,1)
    else:
        X_train_ = X_train
        X_test_ = X_test

    scaler = StandardScaler()
    Y_train_ = Y_train.reshape(-1,1)
    Y_test_ = Y_test.reshape(-1,1)

    scaler.fit(X_train_)
    X_train_ = scaler.transform(X_train_)
    X_test_ = scaler.transform(X_test_)

    scaler.fit(Y_train_)
    Y_train_ = scaler.transform(Y_train_)
    Y_test_ = scaler.transform(Y_test_)

    return X_train_, X_test_, Y_train_, Y_test_




#loading data
cancer = load_breast_cancer()

inputs = cancer.data
targets = cancer.target
labels = cancer.feature_names[0:30]

#Converting to one-hot vectors
X = inputs
y = targets

X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=1/4)

X_train_, X_test_, Y_train_, Y_test_ = scale(X_train, X_test, Y_train, Y_test)
print(X_train.shape)
print(X_train_.shape)
lr = np.linspace(1e-10,1e-2,9)
ep = 100


for lr_ in lr:
    dnn = LinRegClass(X_train_, Y_train, sigmoid, sigmoid_deriv, epochs = ep, eta = lr_, lmbd=0)
    dnn.layers[-1].sigma = sigmoid
    dnn.layers[-1].sigma_d = sigmoid_deriv
    dnn.train(X_train_, Y_train, calcAccuracy=True)
    pred = dnn.predict(X_test_)
    pred = np.ravel(pred[0])

    #acc = dnn.get_ACCtest()
    #print(acc)
"""
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,7))

ax1.scatter(X_test[:,0], pred, label="predi")
ax1.scatter(X_test[:,0], Y_test, label="data", marker="x")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Value")
ax2.scatter(X_test[:,1], pred, label="predi")
ax2.scatter(X_test[:,1], Y_test, label="data", marker="x")
ax2.set_xlabel("Feature 2")
plt.legend()
plt.savefig("Logreg 2 features.png", dpi=300)
plt.show()
"""

#dnn = NeuralNetwork(X_train_, Y_train_, 0, 0, sigmoid, sigmoid_deriv, epochs = ep, eta = 0.001)


"""
#Sigmoid
dnn = NeuralNetwork(X_train_, Y_train_, 2, 16, sigmoid, sigmoid_deriv, epochs = ep, eta = 0.001)
dnn.layers[-1].sigma = linear
dnn.layers[-1].sigma_d = linear_deriv
dnn.train(X_test_, Y_test_, calcMSE = True)
test_predict = dnn.predict(X_test_)



plt.scatter(X_test_, Y_test_, label="Actual", c="r")
plt.scatter(X_test_, test_predict, label="Model", alpha = 0.5)
plt.legend()
plt.savefig("25 ep sigm")
plt.show()
"""

"""
#RELU
dnn1 = NeuralNetwork(X_train_, Y_train_, 2, 16, relu, relu_deriv, epochs = ep, eta = 0.0001)
dnn1.layers[-1].sigma = linear
dnn1.layers[-1].sigma_d = linear_deriv
dnn1.train(X_test_, Y_test_, calcMSE = True)
test_predict = dnn1.predict(X_test_)


#Tanh
dnn2 = NeuralNetwork(X_train_, Y_train_, 2, 16, tanh_function, tanh_deriv, epochs = ep, eta = 0.0001)
dnn2.layers[-1].sigma = linear
dnn2.layers[-1].sigma_d = linear_deriv
dnn2.train(X_test_, Y_test_, calcMSE = True)
test_predict = dnn2.predict(X_test_)

#MSE vs epochs on training
mse = dnn.get_MSEtest()
mse1 = dnn1.get_MSEtest()
mse2 = dnn2.get_MSEtest()

plt.yscale("log")
plt.plot(np.arange(ep), mse, label = "Sigmoid lr: 0.001")
plt.plot(np.arange(ep), mse1, label = "RELU lr: 0.0001")
plt.plot(np.arange(ep), mse2, label = "Tanh lr: 0.0001")
plt.legend()
plt.title(f"Activation funcs : epochs {ep}")
#plt.savefig(f"Act funcs ep_{ep}", dpi=300)
plt.show()
"""
