# imports
import random
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score


# Trivial Model - Always outputs the mean value of the training set
class TrivialSystem(BaseEstimator):
    def __init__(self):
        self.fitted = False
        self.out = None

    def fit(self, X, y):
        self.out = np.mean(y)
        return self

    def predict(self, x_test):
        outputs = np.ones((len(x_test), 1))
        return self.out*outputs

    def score(self, X, y):
        return r2_score(y, self.predict(X))


# Baseline Model - Linear Regression

class MyLinearRegression:
    def __init__(self, X, y, lr, num_epochs=100):
        """
        Linear regression object. The object takes training data(X), training
        outputs(y), learning rate scheduler parameters (A,B) and number of
        epochs as an input
        :param X: np.array
        :param y: np.array
        :param A: float
        :param B: float
        :param num_epochs: int
        """
        self.X = X
        self.y = y
        self.lr = lr
        self.n, self.m = self.X.shape  # n=num examples and m=num features
        self.num_epochs = num_epochs
        # Add one for augmented notation
        self.w = np.random.uniform(low=-0.1, high=0.1, size=self.m + 1)

    def fit(self):
        """
        fit method for the linear regressor to fit the training data using
        mean square regression. The function returns the fitted weights and
        the final loss after linear regression.
        :return: tuple(np.array, list)
        """
        # Augment the data
        self.X = np.append(np.ones((self.n, 1)), self.X, axis=1)
        # Shuffle the data
        shuffle = random.sample(range(len(self.X)), len(self.X))
        self.X = self.X[shuffle]
        self.y = self.y[shuffle]
        # Store the initial error for the halting condition
        _, e_0 = self.predict(self.X, self.y)
        it = 0
        loss = []
        for epoch in range(self.num_epochs):
            for i, point in enumerate(self.X):
                y_predicted = np.dot(self.w, point)
                gradient = (y_predicted - self.y[i]) * point
                self.w = self.w - self.lr * gradient
                it += 1
            _, e_m = self.predict(self.X, self.y)
            loss.append(e_m)
            if e_m < 0.001*e_0:
                print("Early termination!!")
                return self.w, loss
        return self.w, loss

    def predict(self, X_test, y_test=None):
        """
        Predict function to predict y given X. The function also takes an
        optional argument, y_test as input. If y_test is provided,
        the function also returns the loss along with the predictions.
        :param X_test: np.array
        :param y_test: np.array
        :return: np.array, float
        """
        y_predicted = np.dot(X_test, self.w)
        if y_test is not None:
            J_w = np.sum((y_predicted - y_test) ** 2) / len(X_test)
            return y_predicted, np.sqrt(J_w)
        else:
            return y_predicted

