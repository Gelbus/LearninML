from typing import List

import numpy as np

class AdalineGD:
    """
    eta: float
        Скорость обучения 0 - 1
    n_iter: int
        Количество проходов по обучающему набору.
    random_state: int
        Опорное значение генератора случайных чисел для инициализации весов.
    """

    def __init__(self, eta: float, n_iter: int, random_state: int = None) -> None:
        self.eta: float = eta
        self.n_iter: int = n_iter
        self.random_state: float = random_state
        self.losses_: List[float] = []

    def fit(self, X, y):
        rgen: np.random.RandomState = np.random.RandomState(self.random_state)
        self.w_: float = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_: float = np.float32(0.0)


        for i in range (self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2 * errors.mean()
            loss = (errors ** 2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
