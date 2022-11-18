from sklearn.datasets import load_breast_cancer
from NeuralNetwork import *
from Logreg import *
import seaborn as sns
import pandas as pd
from functions import *
import pathlib
import warnings
warnings.filterwarnings("ignore")

colorpal = sns.color_palette("deep")
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes



def calcEtaLambda(X_train_, X_test_, Y_train, Y_test, epochs, act, actDeriv, title):
    eta_vals = np.logspace(-7, 1, 9)
    lmbd_vals = np.logspace(-5, 0, 6)
    lmbd_vals = np.insert(lmbd_vals, 0, 0)

    #Col = learning rate,  Rows = lambdas
    Train_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))      #Define matrices to store accuracy scores as a function
    Test_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))       #of learning rate and number of hidden neurons for

    for i, etaValue in enumerate(eta_vals):
        for j, lmbdValue in enumerate(lmbd_vals):
            dnn = NeuralNetwork(X_train_, Y_train, 2, 16, act, actDeriv, epochs = epochs, etaVal = etaValue, lmbd=lmbdValue)
            dnn.train(X_test_, Y_test, calcAcc=True)
            accTr = dnn.get_accTrain()
            accTe = dnn.get_accTest()
            indexTrain = np.argmax(accTr)
            indexTest = np.argmax(accTe)
            accTra = accTr[indexTrain]
            accTes = accTe[indexTest]
            Train_accuracy[j, i] = accTra
            Test_accuracy[j, i] = accTes

    df = pd.DataFrame(Test_accuracy, columns= eta_vals, index = lmbd_vals)
    df.round(2)
    return df, title


def plotEtaLambda(epochs, savefig=True):
    path = "./Plots/Classification"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    seed = 32455
    np.random.seed(seed)
    #loading data
    cancer = load_breast_cancer()

    inputs = cancer.data
    targets = cancer.target
    labels = cancer.feature_names[0:30]

    #Converting to one-hot vectors
    x = inputs
    y = targets

    #Splitting into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)
    X_train_, X_test_, Y_train_, Y_test_ = scale(X_train, X_test, Y_train, Y_test)

    dfSig, titleSig = calcEtaLambda(X_train_, X_test_, Y_train, Y_test,epochs, sigmoid, sigmoid_deriv, title="Sigmoid")
    dfTanh, titleTanh = calcEtaLambda(X_train_, X_test_, Y_train, Y_test,epochs, tanh, tanh_deriv, title="Tanh")
    dfRelu, titleRelu = calcEtaLambda(X_train_, X_test_, Y_train, Y_test,epochs, relu, relu_deriv, title="Relu")

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16,6), sharey=True, tight_layout=True)
    #fig.tight_layout(rect=[0, 0.1, 1, 0.92])
    plt.rc('axes', titlesize=16)
    plt.subplots_adjust(hspace=0.1)
    plt.suptitle(f"Accuracy Test data w/epochs={epochs}", fontsize = 20, y = 0.05)
    ax1.title.set_text(titleSig)
    ax2.title.set_text(titleTanh)
    ax3.title.set_text(titleRelu)

    ax1 = sns.heatmap(dfSig,  ax=ax1, cbar=False, annot=True, annot_kws={"fontsize":11}, )
    ax2 = sns.heatmap(dfTanh,  ax=ax2, cbar=False, annot=True, annot_kws={"fontsize":11})
    ax3 = sns.heatmap(dfRelu, ax=ax3, cbar=True, annot=True, annot_kws={"fontsize":11})

    axs = [ax1, ax2, ax3]
    ax1.set(ylabel="Lambda")
    for ax in axs:
        ax.set(xlabel="Eta")
    fig.subplots_adjust(wspace=0.001)
    if savefig:
        plt.savefig(f"{path}/TestEtaLamdGrid_{epochs}.pdf", dpi=300)
    #plt.show()



def calcLayerNodes(X_train_, X_test_, Y_train, Y_test, epochs, act, actDeriv, title):
    layer_vals = np.array([5,4,3,2,1])
    #layer_vals = np.array([2,1])
    nodes_vals = np.array([64,32,16,8,4,2])
    #nodes_vals = np.array([32,16])

    #For 300 epochs, manually read best values for eta and lambda for each activation function
    if title == "Sigmoid":
        etaValue = 0.1
        lmbdValue = 0
    elif title == "Tanh":
        etaValue = 0.001
        lmbdValue = 0
    elif title == "Relu":
        etaValue = 0.001
        lmbdValue = 0.0001
    else:
        etaValue = 1e-3
        lmbdValue = 0

    #Col = learning rate,  Rows = lambdas
    Train_accuracy=np.zeros((len(nodes_vals), len(layer_vals)))      #Define matrices to store accuracy scores as a function
    Test_accuracy=np.zeros((len(nodes_vals), len(layer_vals)))       #of learning rate and number of hidden neurons for

    for i, layerValue in enumerate(layer_vals):
        for j, nodeValue in enumerate(nodes_vals):
            dnn = NeuralNetwork(X_train_, Y_train, layerValue, nodeValue, act, actDeriv, epochs = epochs, etaVal = etaValue, lmbd=lmbdValue)
            dnn.train(X_test_, Y_test, calcAcc=True)
            accTr = dnn.get_accTrain()
            accTe = dnn.get_accTest()
            indexTrain = np.argmax(accTr)
            indexTest = np.argmax(accTe)
            accTra = accTr[indexTrain]
            accTes = accTe[indexTest]
            Train_accuracy[j, i] = accTra
            Test_accuracy[j, i] = accTes

    df = pd.DataFrame(Test_accuracy, columns= layer_vals, index = nodes_vals)
    df.round(3)
    return df, title


def plotLayerNodes(epochs, savefig=True):
    path = "./Plots/Classification"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    seed = 32455
    np.random.seed(seed)
    #loading data
    cancer = load_breast_cancer()

    inputs = cancer.data
    targets = cancer.target
    labels = cancer.feature_names[0:30]

    #Converting to one-hot vectors
    x = inputs
    y = targets

    #Splitting into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)
    X_train_, X_test_, Y_train_, Y_test_ = scale(X_train, X_test, Y_train, Y_test)

    dfSig, titleSig = calcLayerNodes(X_train_, X_test_, Y_train, Y_test,epochs, sigmoid, sigmoid_deriv, title="Sigmoid")
    dfTanh, titleTanh = calcLayerNodes(X_train_, X_test_, Y_train, Y_test,epochs, tanh, tanh_deriv, title="Tanh")
    dfRelu, titleRelu = calcLayerNodes(X_train_, X_test_, Y_train, Y_test,epochs, relu, relu_deriv, title="RELU")

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16,6), sharey=True, tight_layout=True)
    #fig.tight_layout(rect=[0, 0.1, 1, 0.92])
    plt.rc('axes', titlesize=16)
    plt.subplots_adjust(hspace=0.1)
    plt.suptitle(f"Accuracy Test data w/epochs={epochs}", fontsize = 20, y = 0.05)
    ax1.title.set_text(titleSig)
    ax2.title.set_text(titleTanh)
    ax3.title.set_text(titleRelu)

    ax1 = sns.heatmap(dfSig,  ax=ax1, cbar=False, annot=True, annot_kws={"fontsize":11}, )
    ax2 = sns.heatmap(dfTanh,  ax=ax2, cbar=False, annot=True, annot_kws={"fontsize":11})
    ax3 = sns.heatmap(dfRelu, ax=ax3, cbar=True, annot=True, annot_kws={"fontsize":11})

    axs = [ax1, ax2, ax3]
    ax1.set(ylabel="Nodes")
    ax1.set(xlabel="Layers")
    ax3.set(xlabel="Layers")
    fig.subplots_adjust(wspace=0.001)
    if savefig:
        plt.savefig(f"{path}/TestLayNodesGrid_{epochs}.pdf", dpi=300)
    plt.show()

def runClassiAcc():
    epochs = 30
    plotEtaLambda(epochs)
    plotLayerNodes(epochs)
    epochs = 300
    plotEtaLambda(epochs)
    plotLayerNodes(epochs)

def runAccTestTrain():
    path = "./Plots/Classification"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    seed = 32455
    np.random.seed(seed)
    #loading data
    cancer = load_breast_cancer()

    inputs = cancer.data #30 features
    targets = cancer.target
    labels = cancer.feature_names[0:30]
    epochs = 250
    etaValue = 1e-3
    lmbdValue = 0
    #Converting to one-hot vectors
    x = inputs
    y = targets

    #Splitting into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)
    X_train_, X_test_, Y_train_, Y_test_ = scale(X_train, X_test, Y_train, Y_test)

    dnn = NeuralNetwork(X_train_, Y_train, 2, 16, sigmoid, sigmoid_deriv, epochs = epochs, etaVal = etaValue, lmbd=lmbdValue)
    dnn.train(X_test_, Y_test, calcAcc=True)
    accTr = dnn.get_accTrain()
    accTe = dnn.get_accTest()

    indexTrain = np.argmax(accTr)
    indexTest = np.argmax(accTe)
    #print(accTr[-1])
    #print(accTe[-1])
    plt.plot(np.arange(epochs), accTr, label = "Train")
    plt.plot(np.arange(epochs), accTe, label = "Test")
    plt.scatter(indexTrain, accTr[indexTrain], marker="x", color = "navy", s=35, label=f"Max acc train {100*accTr[indexTrain]:.1f}%")
    plt.scatter(indexTest, accTe[indexTest], marker="x", color="red", s=35, label=f"Max acc test {100*accTe[indexTest]:.1f}%")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (1=100%)")
    plt.title(f"Test vs Train Accuracy: lr={etaValue}")
    plt.savefig(f"{path}/TestvTrainepochs.pdf", dpi=300)
    #plt.show()

if __name__ == "__main__":
    #runClassiAcc()
    #runAccTestTrain()


"""
for eta in eta_vals
dnn = NeuralNetwork(X_train_, Y_train, 2, 16, sigmoid, sigmoid_deriv, epochs = ep, eta = 1e-3, lmbd=0)
dnn.train(X_test_, Y_test, calcAcc=True)
accTr = dnn.get_accTrain()
accTe = dnn.get_accTest()

indexTrain = np.argmax(accTr)
indexTest = np.argmax(accTe)
print(accTr[-1])
print(accTe[-1])
plt.plot(np.arange(ep), accTr, label = "Train")
plt.plot(np.arange(ep), accTe, label = "Test")
plt.legend()
plt.show()

"""
