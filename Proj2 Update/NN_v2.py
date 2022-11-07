import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Layer:
    def __init__(self, prevLayer, n_nodes, sigma, simga_d, bias=0.1):
        self.n_nodes = n_nodes
        self.prevLayer = prevLayer
        
        if isinstance(prevLayer, Layer):
            self.n_weights = prevLayer.n_nodes
        else:
            self.n_weights = prevLayer

        self.init_weights()
        self.init_bias(bias)

        self.sigma = sigma
        self.sigma_d = simga_d

    def init_weights(self):
        self.weights = np.random.randn(self.n_weights, self.n_nodes)

    def init_bias(self, bias):
        self.bias = np.zeros(self.n_nodes) + bias

    @property
    def get_bias(self):
        return self.bias

    @get_bias.setter
    def get_bias(self, bias):
        self.bias = bias
    
    @property
    def get_weights(self):
        return self.weights

    @get_weights.setter
    def get_weights(self, weights):
        self.weights = weights

    @property
    def get_z(self):
        return self.z

    @get_z.setter
    def get_z(self, z):
        self.z = z

    @property
    def get_a(self):
        return self.a

    @get_a.setter
    def get_a(self, a):
        self.a = a

class NeuralNetwork:
    def __init__(self, X_data, Y_data, n_layers, n_nodes, sigma, sigma_d, epochs=10, batch_size=100, eta=0.1, lmbd=0):
        if len(X_data.shape) == 2:
            self.X_data_full = X_data
        else:
            self.X_data_full = X_data.reshape(-1, 1)
        if len(Y_data.shape) == 2:
            self.Y_data_full = Y_data
        else:
            self.Y_data_full = Y_data.reshape(-1, 1)

        self.n_inputs = self.X_data_full.shape[0]
        self.n_features = self.X_data_full.shape[1]
        self.n_outputs = self.Y_data_full.shape[1]
        # if len(X_data.shape) == 2:
        #     self.n_features = X_data.shape[1]
        # else:
        #     self.n_features = 1

        # if len(X_data.shape) == 2:
        #     self.n_outputs = Y_data.shape[1]
        # else:
        #     self.n_outputs = 1
        



        #initializing layers
        if isinstance(n_nodes, int):
            self.layers = [Layer(self.n_features, n_nodes, sigma, sigma_d)]
            for i in range(1, n_layers+1):
                self.layers.append(Layer(self.layers[i-1], n_nodes, sigma, sigma_d))
            # self.output_layer = Layer(self.layers[i-1], self.n_outputs, sigma, sigma_d)
            self.layers.append(Layer(self.layers[i-1], self.n_outputs, sigma, sigma_d))
        else:
            self.layers = [Layer(self.n_features, n_nodes[0], sigma, sigma_d)]
            for i in n_nodes[1:]:
                self.layers.append(Layer(self.layers[i-1], i, sigma, sigma_d))
            # self.output_layer = Layer(self.layers[i-1], self.n_outputs, sigma, sigma_d)
            self.layers.append(Layer(self.layers[i-1], self.n_outputs, sigma, sigma_d))


        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd


    def feedForward(self):
        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias

        #print(np.shape(bias), np.shape(weights), np.shape(self.X_data))
        #z = np.matmul(weights, self.X_data) + bias
        # z = np.matmul(self.X_data, weights) + bias
        # a = [z]
        # layer1.get_a = z
        # for layer in self.layers[1:]:
        #     layer.get_z = z
        #     al = layer.sigma(z)
        #     layer.get_a = al
        #     a.append(al)
        #     z = al

        z = np.matmul(self.X_data, weights) + bias
        layer1.get_z = z
        layer1.get_a = layer1.sigma(z)
        a = [layer1.get_a]

        for layer in self.layers[1:]:
            z = np.matmul(a[-1], layer.get_weights) + layer.get_bias
            layer.get_z = z
            layer.get_a = layer.sigma(z)
            a.append(layer.get_a)

        self.output = a[-1]
        #self.a = np.array(a)

    def feedForwardOut(self, X):
        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias

        # print(np.shape(bias), np.shape(weights), np.shape(X.T))
        # z = np.matmul(X.T, weights) + bias
        # print(z.shape, np.matmul(X.T, weights).shape)
        # a = [z]
        # layer1.get_a = z
        # for layer in self.layers[1:]:
        #     layer.get_z = z
        #     al = layer.sigma(z)
        #     layer.get_a = al
        #     a.append(al)
        #     z = al

        z = np.matmul(X, weights) + bias
        print(z.shape, weights.shape, bias.shape)
        layer1.get_z = z
        layer1.get_a = layer1.sigma(z)
        a = [layer1.get_a]

        for layer in self.layers[1:]:
            weights = layer.get_weights
            bias = layer.get_bias
            z = np.matmul(a[-1], weights) + bias
            print(z.shape, weights.shape, bias.shape)
            layer.get_z = z
            layer.get_a = layer.sigma(z)
            a.append(layer.get_a)

        return z

    def backProp(self):
        Y_data = self.Y_data

        error_output = self.output - Y_data
        error = [error_output]

        #going through backwards
        for layer in self.layers[:-1:-1]:
            error.append(np.matmul(error[-1], layer.prevLayer.get_weights.T)*layer.sigma_d(layer.get_z))
            #error.append(np.sum(error[-1]*layer.prevLayer.get_weights*layer.sigma_d(layer.get_z))) #, axis=?
            #weights_grad = self.eta*np.matmul(error[-1].T, layer.prevLayer.get_a)
            #or
            weights_grad = self.eta*np.matmul(layer.prevLayer.get_a.T, error[-1])
            layer.get_weights = layer.get_weights - weights_grad
            layer.get_bias = layer.get_bias - self.eta*error[-1]

        
    def train(self):
        data_indices = np.arange(self.n_inputs)
        
        for i in range(self.epochs):
            for j in range(self.iterations):
                #(750,) 100
                #print(data_indices.shape, self.batch_size)
                chosen_data_points = np.random.choice(data_indices, size=self.batch_size, replace=False)

                self.X_data = self.X_data_full[chosen_data_points]
                self.Y_data = self.Y_data_full[chosen_data_points]

                self.feedForward()
                self.backProp()

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        output = self.feedForwardOut(X)
        return output


n = 1000
x = np.linspace(0, 10, n)

def f(x):
    return 1 + 5*x + 3*x**2 

y = f(x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
    sig_x  = sigmoid(x)
    return sig_x*(1 - sig_x)

X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)

dnn = NeuralNetwork(X_train, Y_train, 2, 40, sigmoid, sigmoid_deriv)

dnn.train()

test_predict = dnn.predict(X_test)
print(test_predict.shape)
print(Y_test.shape)
plt.scatter(X_test, Y_test, label="Actual", c="r")
plt.scatter(X_test, test_predict, label="Model")
plt.legend()
plt.show()