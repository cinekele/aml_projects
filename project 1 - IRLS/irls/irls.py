import abc

import numpy as np
from scipy.special import expit  # sigmoid function


class Optimizer(abc.ABC):
    def step_opt(self, weights, X, y):
        pass


class GradientDescent(Optimizer):
    __slots__ = ["alpha", "reg_term"]

    def __init__(self, alpha, reg_term, offset):
        self.alpha = alpha
        self.reg_term = reg_term
        self.offset = offset

    def step_opt(self, weights, X, y):
        n = X.shape[0]
        p = expit(X @ weights)
        loss = p - y

        without_bias = np.copy(weights)
        without_bias[0] = 0  # don't penalize bias

        grad = self.alpha * np.dot(loss.T, X).T / n  # real gradient
        reg_grad = self.reg_term * without_bias / n  # regularization term

        new_weights = weights - (grad + reg_grad)

        jcost = (-1 / n) * (
                np.dot(np.log(p + self.offset).T, y) + np.dot(np.log((np.ones(n) - p + self.offset)).T, np.ones(n) - y))
        jreg = self.reg_term / (2 * n) * np.dot(without_bias.T, without_bias)
        j_new = jcost + jreg

        return new_weights, j_new


class IRLS(Optimizer):
    __slots__ = ["offset"]

    def __init__(self, offset):
        self.offset = offset

    def step_opt(self, weights, X, y):
        n = X.shape[0]
        p = expit(np.dot(X, weights))
        loss = y - p
        w = np.diag(p * (1 - p) + self.offset)
        inverse_w = np.linalg.inv(w)
        z = X @ weights + inverse_w @ loss
        weights_new = np.linalg.inv(X.T @ w @ X) @ X.T @ w @ z

        jcost = (-1 / n) * (
                np.dot(np.log(p + self.offset).T, y) + np.dot(np.log((np.ones(n) - p + self.offset)).T, np.ones(n) - y))
        return weights_new, jcost


class LogisticRegression:
    __slots__ = ["_theta", "optimizer", "max_num_iters", "tol"]

    def __init__(self, optimizer: str = "IRLS", max_num_iters: int = 150, tol=1e-4, **kwargs):
        """
        Initialization of class
        :param optimizer: optimizer of logistic regression
        :type optimizer: str
        :param max_num_iters: max number of iteration of gradient descent
        :type max_num_iters: int
        """
        self._theta = None
        self.max_num_iters = max_num_iters
        self.tol = tol
        offset = kwargs["offset"] if "offset" in kwargs else 1e-6
        if optimizer == "GB":
            alpha = kwargs["alpha"] if "alpha" in kwargs else 1e-4
            reg_term = kwargs["reg_term"] if "reg_term" in kwargs else 0.0001
            self.optimizer = GradientDescent(alpha, reg_term, offset)
        elif optimizer == "IRLS":
            self.optimizer = IRLS(offset)
        else:
            raise Exception("This optimizer doesn't exist")

    @property
    def coefficients(self):
        """
        :returns: coefficients of properities
        :rtype: np.double
        """
        return np.copy(self._theta[1:])

    @property
    def bias(self):
        """
        :return: intercept
        :rtype: np.double
        """
        return np.copy(self._theta[0])

    def fit(self, X: np.array, y: np.array) -> "LogisticRegression":
        """
        Train logistic regression using gradient decent
        :param X: learning examples
        :type X: np.array
        :param y: target
        :type y: np.array
        """
        n = X.shape[0]
        p = X.shape[1]
        self._theta = np.zeros(p + 1)
        j = np.finfo(np.float64).max  # cost function

        X_with_ones = np.concatenate((np.ones((n, 1)), X), axis=1)

        for i in range(self.max_num_iters):
            new_weights, j_new = self.optimizer.step_opt(self._theta, X_with_ones, y)
            self._theta = new_weights
            if abs(j_new - j) < self.tol:  # stop można zmienić
                break
            j = j_new
        return self

    def predict(self, X: np.array, threshold: float = 0.5) -> np.array:
        """
        Predict based on given values
        :type X: np.double
        :param X: predicted examples
        :type threshold: float
        :param threshold: threshold
        :returns: predicted value
        :rtype: np.double
        """
        n = X.shape[0]
        X_with_ones = np.concatenate((np.ones((n, 1)), X), axis=1)
        return expit(np.dot(X_with_ones, self._theta)) >= threshold

    def predict_proba(self, X: np.array) -> np.array:
        """
        :type X: np.double
        :param X: predicted examples
        :returns: probabilities of values
        :rtype: np.double
        """
        n = X.shape[0]
        X_with_ones = np.concatenate((np.ones((n, 1)), X), axis=1)
        return expit(np.dot(X_with_ones, self._theta))
