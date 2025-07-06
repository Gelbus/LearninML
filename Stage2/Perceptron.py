from typing import List

import numpy as np

class Perceptron:
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

    def fit(self, X, y):
        rgen: np.random.RandomState = np.random.RandomState(self.random_state)
        self.w_: float = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_: float = np.float32(0.0)
        self.errors_: List[int] = []

        for _ in range(self.n_iter):
            errors: int = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
