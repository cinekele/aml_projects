import abc

import numpy as np
import pandas as pd

from scipy.special import expit  # sigmoid function
from typing import List
from sklearn.base import BaseEstimator


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

        p = expit(X @ new_weights)
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
        w = p * (1 - p) + self.offset
        inverse_w = 1 / w
        z = X @ weights + inverse_w * loss
        weights_new = np.linalg.inv(X.T * w @ X) @ X.T * w @ z

        p = expit(X @ weights_new)
        jcost = (-1 / n) * (
                np.dot(np.log(p + self.offset).T, y) + np.dot(np.log((np.ones(n) - p + self.offset)).T, np.ones(n) - y))
        return weights_new, jcost


class LogisticRegression(BaseEstimator):
    __slots__ = ["_theta", "optimizer", "max_num_iters", "tol", "_interactions", "_theta_hist"]

    def __init__(self, optimizer: str = "IRLS", max_num_iters: int = 1000, tol=1e-6, **kwargs):
        """
        Initialization of class
        :param optimizer: optimizer of logistic regression
        :type optimizer: str
        :param max_num_iters: max number of iteration of gradient descent
        :type max_num_iters: int
        :param tol: parameter, which indicate when theta converges
        :type tol: float
        """
        self._theta = None
        self._theta_hist = None
        self._interactions = None
        self.max_num_iters = max_num_iters
        self.tol = tol
        offset = kwargs["offset"] if "offset" in kwargs else 1e-4
        if optimizer == "GB":
            alpha = kwargs["alpha"] if "alpha" in kwargs else 1e-4
            reg_term = kwargs["reg_term"] if "reg_term" in kwargs else 0.0001
            self.optimizer = GradientDescent(alpha, reg_term, offset)
        elif optimizer == "IRLS":
            self.optimizer = IRLS(offset)
        elif issubclass(optimizer.__class__, Optimizer):
            self.optimizer = optimizer
        else:
            raise Exception("This optimizer doesn't exist")

    @property
    def coefficients(self) -> np.array:
        """
        Get coefficients of logistic regression
        :returns: coefficients of properities
        :rtype: np.array
        """
        return np.copy(self._theta[1:])

    @property
    def bias(self) -> np.array:
        """
        Get bias of logistic regression
        :return: intercept
        :rtype: np.array
        """
        return np.copy(self._theta[0])

    @property
    def theta(self) -> np.array:
        """
        Get whole theta
        :return: theta
        :rtype: np.array
        """
        return np.copy(self._theta)

    @property
    def interactions(self) -> np.array:
        """
        Get matrix of interactions included in model
        :return: interactions
        :rtype: np.array
        """
        return np.copy(self._interactions)

    def theta_history(self) -> List[np.array]:
        """
        Return history of changes theta parameter
        :return: theta_history
        :rtype: list[np.array]
        """
        return self._theta_hist

    def fit(self, X: np.array, y: np.array, interactions: np.array = None) -> "LogisticRegression":
        """
        Train logistic regression
        :param interactions: 2 column matrix with variable pairs to include as interactions
        :type interactions: np.array
        :param X: learning examples
        :type X: np.array
        :param y: target
        :type y: np.array
        """
        if interactions is not None:
            self._set_interactions(interactions)
        X = self._include_interactions(X)
        n = X.shape[0]
        p = X.shape[1]
        self._theta = np.zeros(p + 1)
        self._theta_hist = [np.copy(self._theta)]
        j = np.finfo(np.float64).max  # cost function

        X_with_ones = np.concatenate((np.ones((n, 1)), X), axis=1)

        for i in range(self.max_num_iters):
            new_weights, j_new = self.optimizer.step_opt(self._theta, X_with_ones, y)
            self._theta = new_weights
            self._theta_hist += [np.copy(new_weights)]
            if abs(j_new - j) < self.tol:  # TODO: Zmienić na skleanra, czyli jeśli gradient w którymś kierunku jest mniejszy od tola
                break
            j = j_new
        return self

    def _set_interactions(self, interactions):
        if isinstance(interactions, pd.DataFrame):
            interactions = interactions.to_numpy()
        self._interactions = interactions

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
        return self.predict_proba(X) >= threshold

    def predict_proba(self, X: np.array) -> np.array:
        """
        :type X: np.double
        :param X: predicted examples
        :returns: probabilities of values
        :rtype: np.double
        """
        n = X.shape[0]
        X_interactions = self._include_interactions(X)
        X_with_ones = np.concatenate((np.ones((n, 1)), X_interactions), axis=1)
        return expit(np.dot(X_with_ones, self._theta))

    def _include_interactions(self, X):
        if self._interactions is None:
            return X
        if isinstance(X, pd.DataFrame):
            new_cols = np.stack([X[pair[0]] * X[pair[1]] for pair in self._interactions], axis=1)
        else:
            new_cols = np.stack([X[:, pair[0]] * X[:, pair[1]] for pair in self._interactions], axis=1)
        return np.hstack([X, new_cols])
