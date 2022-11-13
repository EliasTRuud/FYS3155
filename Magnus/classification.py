from sklearn.datasets import load_breast_cancer
from classes import *

print("Loading data")
cancer = load_breast_cancer()

inputs = cancer.data
targets = cancer.target
labels = cancer.feature_names[0:30]

print("Converting to one-hot vectors")
x = inputs
y = to_categorical_numpy(targets)

print("Splitting into train and test data")
X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)



n_neuron = 32
eta = 0.001

# n_neuron = np.logspace(0,3,4,dtype=int)
# eta = np.logspace(-3,-1,3)
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)

Train_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))      #Define matrices to store accuracy scores as a function
Test_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))       #of learning rate and number of hidden neurons for 

print("Starting training")
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = NeuralNetwork(X_train, Y_train, 1, 32, sigmoid, sigmoid_deriv, epochs = 100, eta = eta, lmbd=lmbd)
        dnn.layers[-1].sigma = softmax
        dnn.train()
        Train_accuracy[i, j] = dnn.evaluate(X_train, Y_train) #problem
        Test_accuracy[i, j] = dnn.evaluate(X_test, Y_test)

print("Finished traning")
plot_data(eta_vals, lmbd_vals, Train_accuracy, 'Training')
plot_data(eta_vals, lmbd_vals, Test_accuracy, 'Testing')

# plt.scatter(X_test, Y_test, label="Actual", c="r")
# plt.scatter(X_test, test_predict, label="Model", alpha = 0.5)
# plt.scatter(X_test, test_predict_untrained, label="Model_none")
# plt.legend()
# plt.show()