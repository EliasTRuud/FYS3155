from sklearn.linear_model import LinearRegression as OLS_reg
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
import copy

colorpal = sns.color_palette("deep")
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes


def CostOLS(y, X, beta):
    return (1/y.shape[0])*((y- X@beta).T)@(y- X@beta)

def CostRidge(y, X, beta, lambda_):
    return (1/y.shape[0])*((y- X@beta).T)@(y- X@beta) + lambda_*beta.T@beta

#Define the gradient with costfunction
gradientOLS = grad(CostOLS, 2) #2 meaning beta
gradientRidge =  grad(CostRidge, 2) #2 meaning beta

def gradient_decent(X, x, y, beta, lr, n_iter, momentum=0, batch_size=1, useAda=False, useRMS=False, useAdam=False, lambda_ = 0):
    """
    Performs gradient_decent on x dataset with y being target data. Option of 3 different optimizers
    and uses Ridge if lambda value differs from 0. Returns MSE and estimated beta values.
    """
    beta = copy.deepcopy(beta) #avoid overwriting in original beta
    MSE_list = [] #Store MSE scores every update to plot
    beta_list = [] #Store every time beta is updated
    change = 0
    M = batch_size
    m = int(n_iter/M) #number of minibatches
    y_pred = beta[0] + beta[1]*x + beta[2]*x*x
    MSE_list.append(mse(y, y_pred))
    beta_list.append(beta)

    if useAda:
        delta = 1e-8
        for i in range(n_iter):
            Giter = np.zeros(shape=(3,3))
            for i in range(m):
                #Split up X and y
                rand_ind = M*np.random.randint(m)
                xi = X[rand_ind:rand_ind+M]
                yi = y[rand_ind:rand_ind+M]

                if lambda_ != 0:
                    gradients = gradientRidge(yi, xi, beta, lambda_)
                else:
                    gradients = gradientOLS(yi, xi, beta)

                # Calculate the outer product of the gradients
                Giter +=gradients @ gradients.T
                # Simpler algorithm with only diagonal elements
                Ginverse = np.c_[lr/(delta+np.sqrt(np.diagonal(Giter)))]
                # compute update
                update = np.multiply(Ginverse,gradients)

                new_change = update + momentum*change
                beta -= new_change
                change = new_change
                y_pred = beta[0] + beta[1]*x + beta[2]*x*x
                MSE_list.append(mse(y, y_pred))
                beta_list.append(beta)
                #Calculate MSE and store to list and plot.
                # momentum vs non-momentum
    elif useRMS:
        rho = 0.99
        lr = 0.01
        delta = 1e-8
        for i in range(n_iter):
            Giter = np.zeros(shape=(3,3))
            for i in range(m):
                #Split up X and y
                rand_ind = M*np.random.randint(m)
                xi = X[rand_ind:rand_ind+M]
                yi = y[rand_ind:rand_ind+M]
                if lambda_ != 0:
                    gradients = gradientRidge(yi, xi, beta, lambda_)
                else:
                    gradients = gradientOLS(yi, xi, beta)

                # Previous value for the outer product of gradients
                Previous = Giter
        	    # Accumulated gradient
                Giter +=gradients @ gradients.T
        	    # Scaling with rho the new and the previous results
                Gnew = (rho*Previous+(1-rho)*Giter)
        	    # Taking the diagonal only and inverting
                Ginverse = np.c_[lr/(delta+np.sqrt(np.diagonal(Gnew)))]
        	    # Hadamard product
                update = np.multiply(Ginverse,gradients)

                new_change = update + momentum*change
                beta -= new_change
                change = new_change
                y_pred = beta[0] + beta[1]*x + beta[2]*x*x
                MSE_list.append(mse(y, y_pred))
                beta_list.append(beta)
    elif useAdam:
        b1 = 0.9
        b2 = 0.999
        t = 0
        eps = 1e-8
        m_ = 0
        v = 0
        for i in range(n_iter):
            for i in range(m):
                rand_ind = M*np.random.randint(m)
                xi = X[rand_ind:rand_ind+M]
                yi = y[rand_ind:rand_ind+M]

                t = t + 1

                if lambda_ != 0:
                    gradients = gradientRidge(yi, xi, beta, lambda_)
                else:
                    gradients = gradientOLS(yi, xi, beta)

                m_ = b1*m_ + (1-b1)*gradients
                v = b2*v + (1-b2)*gradients**2
                m_hat = m_/(1-b1**t)
                v_hat = v/(1-b2**t)
                update = lr*m_hat/(np.sqrt(v_hat)+eps)

                new_change = lr*update + momentum*change
                beta -= new_change
                change = new_change
                y_pred = beta[0] + beta[1]*x + beta[2]*x*x
                MSE_list.append(mse(y, y_pred))
                beta_list.append(beta)
    else:
        for i in range(n_iter):
            for i in range(m):
                #Split up X and y
                rand_ind = M*np.random.randint(m)
                xi = X[rand_ind:rand_ind+M]
                yi = y[rand_ind:rand_ind+M]

                if lambda_ != 0:
                    gradients = gradientRidge(yi, xi, beta, lambda_)
                else:
                    gradients = gradientOLS(yi, xi, beta)

                new_change = lr*gradients + momentum*change
                beta -= new_change
                change = new_change
                y_pred = beta[0] + beta[1]*x + beta[2]*x*x
                MSE_list.append(mse(y, y_pred))
                beta_list.append(beta)
                #Calculate MSE and store to list and plot.
                # momentum vs non-momentum
    #plt.plot(MSE_list)
    #plt.show()
    return beta, MSE_list, beta_list

def calcSGDGridsearch(X, x, y):
    eta_vals = np.logspace(-7, 1, 9)
    lmbd_vals = np.logspace(-5, 0, 6)
    lmbd_vals = np.insert(lmbd_vals, 0, 0)

    #Col = learning rate,  Rows = lambdas
    Train_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))      #Define matrices to store accuracy scores as a function
    Test_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))       #of learning rate and number of hidden neurons for

    for i, etaValue in enumerate(eta_vals):
        for j, lmbdValue in enumerate(lmbd_vals):
            gradient_decent(X, x, y, beta_ada, lr=etaValue, n_epochs, batch_size=M, useAda=True, lambda_=lmbdValue)
            dnn.train(X_test_, Y_test, calcAcc=True)
            accTr = dnn.get_accTrain()
            accTe = dnn.get_accTest()
            indexTrain = np.argmax(accTr)
            indexTest = np.argmax(accTe)
            accTra = accTr[indexTrain]
            accTes = accTe[indexTest]
            Train_accuracy[j, i] = accTra
            Test_accuracy[j, i] = accTes



def runPlotsSGD():
    n = 10000
    x = np.random.rand(n,1)
    #Analytical value. Static learning rate.
    y = 2+3*x+4*x*x+0.1*np.random.rand(n,1)*0.1

    X = np.c_[np.ones((n,1)), x, x**2] #design matrix
    XT_X = X.T @ X
    #theta_linreg = np.linalg.pinv(XT_X) @ (X.T @ y)
    H = (2.0/n)* XT_X
    EigValues, EigVectors = np.linalg.eig(H)
    lr = 1.0/np.max(EigValues)

    #X_train, X_test, y_train, y_test = train_test_split(X,z, test_size=0.2)
    #Simple OLS fit
    model = OLS_reg(fit_intercept=True)
    model.fit(X, y)
    y_pred = model.intercept_ + model.coef_[0][1]*x + model.coef_[0][2]*x*x

    np.random.seed(200)
    beta = np.random.randn(3,1)
    np.random.seed(200)
    beta_ada = np.random.randn(3,1)
    np.random.seed(200)
    beta_rms = np.random.randn(3,1)
    np.random.seed(200)
    beta_adam = np.random.randn(3,1)

    print(beta)
    #lr = 0.01
    n_epochs = 120
    M = 5   #size of each minibatch

    lr, lambda_ = calcSGDGridsearch()

    beta_, MSE_list, _ = gradient_decent(X, x, y, beta, lr, n_epochs, batch_size=M)
    beta_ada, MSE_list_ada, _ = gradient_decent(X, x, y, beta_ada, lr, n_epochs, batch_size=M, useAda=True, lambda_=0)
    beta_rms, MSE_list_rms, _ = gradient_decent(X, x, y, beta_rms, lr, n_epochs, batch_size=M, useRMS=True, lambda_=0)
    beta_adam, MSE_list_adam, _ = gradient_decent(X, x, y, beta_adam, lr, n_epochs, batch_size=M, useAdam=True, lambda_=0)

    print(beta)
    exit()
    path = "./Plots/GradDescent"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8,6))
    plt.tight_layout()
    plt.title("MSE for different optimizers")
    plt.plot(MSE_list, label="Default")
    plt.plot(MSE_list_ada, label="Ada")
    plt.plot(MSE_list_rms, label="RMS")
    plt.plot(MSE_list_adam, label="Adam")
    plt.legend()
    plt.savefig(f"{path}/mse_opti.pdf", dpi=300)
    plt.show()


    """
    plt.plot(beta, label="Default")
    plt.plot(beta_ada, label="Ada")
    plt.plot(beta_rms, label="RMS")
    plt.legend()
    """




    y_pred_grad = beta[0] + beta[1]*x + beta[2]*x*x
    y_pred_grad_ada = beta_ada[0] + beta_ada[1]*x + beta_ada[2]*x*x
    y_pred_grad_rms = beta_rms[0] + beta_rms[1]*x + beta_rms[2]*x*x
    y_pred_grad_adam = beta_adam[0] + beta_adam[1]*x + beta_adam[2]*x*x

    plt.figure(figsize=(8,6))
    plt.tight_layout()
    plt.title("Data vs predicted data w/optimizers")
    plt.plot(x, y,".", label="Data")
    plt.plot(x, y_pred_grad, ".", label="Default")
    plt.plot(x, y_pred_grad_ada, ".", label="Ada opti")
    plt.plot(x, y_pred_grad_rms, ".", label="RMS opti")
    plt.plot(x, y_pred_grad_adam, ".", label="Adam opti")
    plt.legend()
    plt.savefig(f"{path}/target_vs_pred.pdf", dpi=300)
    plt.show()


    #NOTES:
    # Opg a) juster hyperparametere slik at vi klart ser forskjell på performance.
    # Prøv ulike learning rates, statisk.
    # Adam : https://arxiv.org/abs/1412.6980

if __name__ == "__main__":
    runPlotsSGD()
