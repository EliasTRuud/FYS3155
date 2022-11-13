from classes import *
from sklearn import datasets

n = 10000
x = np.linspace(0, 3, n)

y = np.copy(x)
z = FrankeFunction(x, y)

X = np.array([x, y]).T
print(X.shape, z.shape)

# X = np.c_[np.ones((n,1)), x, x**2]
y = f(x)
X_train, X_test, Y_train, Y_test = train_test_split(X, z,test_size=1/4)

X_train_, X_test_, y_mean_train = scale(X_train, X_test, Y_train)


dnn = NeuralNetwork(X_train, Y_train, 3, 32, tanh, tanh_deriv, epochs = 1000, eta = 0.0001)
dnn.layers[-1].sigma = linear
dnn.layers[-1].sigma_d = linear_deriv

test_predict_untrained = dnn.predict(X_test)
dnn.train()

test_predict = dnn.predict(X_test)

print(test_predict.shape)
print(Y_test.shape)
plt.scatter(X_test[:, 1], Y_test, label="Actual", c="r")
plt.scatter(X_test[:, 1], test_predict[:, 0], label="Model")
# plt.scatter(X_test, test_predict_1[:, 0], label="Model")
# plt.scatter(X_test[:, 1], test_predict_1[:, 0], label="Model_untrained", marker="x", alpha=0.2)
# plt.scatter(X_test, Y_test, label="Actual", c="r")
# plt.scatter(X_test_, test_predict, label="Model", alpha = 0.5)
#plt.scatter(X_test, test_predict_untrained, label="Model_none")
plt.legend()
plt.show()
