from sklearn.datasets import load_breast_cancer
from classes import *

#loading data
cancer = load_breast_cancer()

inputs = cancer.data
targets = cancer.target
labels = cancer.feature_names[0:30]

#Converting to one-hot vectors
x = inputs
y = to_categorical_numpy(targets)

#Splitting into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)



n_neuron = 32

#learning rate and regularization parameters
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)

Train_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))      #Define matrices to store accuracy scores as a function
Test_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))       #of learning rate and number of hidden neurons for 

#Starting training, searching for best combination
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = NeuralNetwork(X_train, Y_train, 3, n_neuron, tanh, tanh_deriv, epochs = 25, eta = eta, lmbd=lmbd, Type="Classification")
        #setting output activation to softmax
        dnn.layers[-1].sigma = softmax
        dnn.train()
        #finding the accuracy
        Train_accuracy[i, j] = dnn.evaluate(X_train, Y_train)
        Test_accuracy[i, j] = dnn.evaluate(X_test, Y_test)

#plotting
plot_data(eta_vals, lmbd_vals, Train_accuracy, 'tanh')
plot_data(eta_vals, lmbd_vals, Test_accuracy, 'tanh')