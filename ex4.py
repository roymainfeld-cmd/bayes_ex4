import numpy as np
from typing import Callable
from matplotlib import pyplot as plt

KERNEL_STRS = {
    'Laplacian': r'Laplacian, $\alpha={}$, $\beta={}$',
    'RBF': r'RBF, $\alpha={}$, $\beta={}$',
    'Spectral': r'Spectral, $\alpha={}$, $\beta={}$, $\gamma={}$',
    'NN': r'NN, $\alpha={}$, $\beta={}$'
}


def average_error(pred: np.ndarray, vals: np.ndarray):
    """
    Calculates the average squared error of the given predictions
    :param pred: the predicted values
    :param vals: the true values
    :return: the average squared error between the predictions and the true values
    """
    return np.mean((pred - vals)**2)


def RBF_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the RBF kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        l2_squared = np.sum((x - y)**2)
        return alpha * np.exp(-beta * l2_squared)
    return kern


def Laplacian_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Laplacian kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        l1_distance = np.sum(np.abs(x - y))
        return alpha * np.exp(-beta * l1_distance)
    return kern


def Spectral_kernel(alpha: float, beta: float, gamma: float) -> Callable:
    """
    An implementation of the Spectral kernel (see https://arxiv.org/pdf/1302.4245.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        diff = x - y
        l2_squared = np.sum(diff**2)
        return alpha * np.exp(-beta * l2_squared) * np.cos(np.pi * diff / gamma)
    return kern


def NN_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Neural Network kernel (see section 4.2.3 of http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        return alpha * (2.0 / np.pi) * np.arcsin(
            (2.0 * beta * (x * y + 1.0)) /
            np.sqrt((1.0 + 2.0 * beta * (1.0 + x**2)) *
                    (1.0 + 2.0 * beta * (1.0 + y**2)))
        )
    return kern


class GaussianProcess:

    def __init__(self, kernel: Callable, noise: float):
        """
        Initialize a GP model with the specified kernel and noise
        :param kernel: the kernel to use when fitting the data/predicting
        :param noise: the sample noise assumed to be added to the data
        """
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.L = None
        self.alpha = None

    def fit(self, X, y) -> 'GaussianProcess':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        self.X_train = X
        self.y_train = y

        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j])

        Ky = K + self.noise * np.eye(n)
        self.K_inv = np.linalg.inv(Ky)
        self.alpha = self.K_inv @ y       
        return self
        
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the MMSE regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        n_test = len(X)
        n_train = len(self.X_train)

        K_star = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                K_star[i, j] = self.kernel(X[i], self.X_train[j])

        return K_star @ self.alpha
        


    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def sample(self, X) -> np.ndarray:
        """
        Sample a function from the posterior
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the sample (same shape as X)
        """
        n_test = len(X)
        n_train = len(self.X_train)

        K_star = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                K_star[i, j] = self.kernel(X[i], self.X_train[j])

        K_star_star = np.zeros((n_test, n_test))
        for i in range(n_test):
            for j in range(n_test):
                K_star_star[i, j] = self.kernel(X[i], X[j])

        mean = K_star @ self.alpha
        cov = K_star_star - K_star @ self.K_inv @ K_star.T

        cov += 1e-8 * np.eye(n_test) # for numerical stability

        return np.random.multivariate_normal(mean, cov)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        n_test = len(X)
        n_train = len(self.X_train)

        K_star = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                K_star[i, j] = self.kernel(X[i], self.X_train[j])

        K_star_star_diag = np.zeros(n_test)
        for i in range(n_test):
            K_star_star_diag[i] = self.kernel(X[i], X[i])

        tmp = K_star @ self.K_inv
        var_f = K_star_star_diag - np.sum(tmp * K_star, axis=1)

        return np.sqrt(var_f)

    def log_evidence(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the model's log-evidence under the training data
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the log-evidence of the model under the data points
        """
        self.fit(X, y)

        n = len(X)

        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j])
        Ky = K + self.noise * np.eye(n)

        term1 = -0.5 * (y.T @ self.K_inv @ y)

        sign, logdet = np.linalg.slogdet(Ky)
        if sign <= 0:
            return -np.inf 

        term2 = -0.5 * logdet

        term3 = -0.5 * n * np.log(2 * np.pi)

        return term1 + term2 + term3


def main():
    # ------------------------------------------------------ section 2.1
    xx = np.linspace(-5, 5, 500)
    x, y = np.array([-2, -1, 0, 1, 2]), np.array([2.4, .9, 2.8, -2.9, -1.5])

    # ------------------------------ questions 2 and 3
    # choose kernel parameters
    params = [
        # Laplacian kernels
        ['Laplacian', Laplacian_kernel, 1, 0.25],           # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, None, None],        # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, None, None],        # insert your parameters, order: alpha, beta

        # RBF kernels
        ['RBF', RBF_kernel, 1, 0.25],                       # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, None, None],                    # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, None, None],                    # insert your parameters, order: alpha, beta

        # Gibbs kernels
        ['Spectral', Spectral_kernel, 1, .5, 3],            # insert your parameters, order: alpha, beta, gamma
        ['Spectral', Spectral_kernel, None, None, None],    # insert your parameters, order: alpha, beta, gamma
        ['Spectral', Spectral_kernel, None, None, None],    # insert your parameters, order: alpha, beta, gamma

        # Neurel network kernels
        ['NN', NN_kernel, 1, 0.25],                         # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, None, None],                      # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, None, None],                      # insert your parameters, order: alpha, beta
    ]
    noise_var = 0.05

    # plot all of the chosen parameter settings
    for p in params:
        # create kernel according to parameters chosen
        k = p[1](*p[2:])    # p[1] is the kernel function while p[2:] are the kernel parameters

        # initialize GP with kernel defined above
        gp = GaussianProcess(k, noise_var)

        # plot prior variance and samples from the priors
        plt.figure()
        # todo <your code here>
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])

        # fit the GP to the data and calculate the posterior mean and confidence interval
        gp.fit(x, y)
        m, s = gp.predict(xx), 2*gp.predict_std(xx)

        # plot posterior mean, confidence intervals and samples from the posterior
        plt.figure()
        plt.fill_between(xx, m-s, m+s, alpha=.3)
        plt.plot(xx, m, lw=2)
        for i in range(6): plt.plot(xx, gp.sample(xx), lw=1)
        plt.scatter(x, y, 30, 'k', zorder=10)
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])
        plt.show()

    # ------------------------------ question 4
    # define range of betas
    betas = np.linspace(0.1, 7, 101)
    noise_var = .27

    # calculate the evidence for each of the kernels
    evidence = [GaussianProcess(RBF_kernel(1, beta=b), noise_var).log_evidence(x, y) for b in betas]

    # plot the evidence as a function of beta
    plt.figure()
    plt.plot(betas, evidence, lw=2)
    plt.xlabel(r'$\beta$')
    plt.ylabel('log-evidence')
    plt.show()

    # extract betas that had the min, median and max evidence
    srt = np.argsort(evidence)
    min_ev, median_ev, max_ev = betas[srt[0]], betas[srt[(len(evidence)+1)//2]], betas[srt[-1]]

    # plot the mean of the posterior of a GP using the extracted betas on top of the data
    plt.figure()
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=min_ev), noise_var).fit(x, y).predict(xx), lw=2, label='min evidence')
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=median_ev), noise_var).fit(x, y).predict(xx), lw=2, label='median evidence')
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=max_ev), noise_var).fit(x, y).predict(xx), lw=2, label='max evidence')
    plt.scatter(x, y, 30, 'k', alpha=.5)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()



