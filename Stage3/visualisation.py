from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_errors(err: List[int]) -> None:
    plt.plot(range(1, len(err) + 1), err, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Update count')
    plt.show()


def plot_decision_regions(X: np.ndarray, y: np.ndarray, classifier: Any, test_idx=None, resolution: float = 0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black', alpha=1,
                    linewidths=1, marker='o',
                    s=100, label='Test set')

    plt.xlabel('Длина депестка norm')
    plt.ylabel('Ширина лепестка norm')
    plt.legend(loc='upper left')
    plt.show()


# def plot_losses(classifier1: Any, classifier2: Any) -> None:
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
#     ax[0].plot(range(1, len(classifier1.losses_) + 1), np.log10(classifier1.losses_), marker='o')
#     ax[0].set_xlabel('Epoch')
#     ax[0].set_ylabel('log(loss)')
#     ax[0].set_title(f'Adaline - learning speed {classifier1.eta}')
#
#     ax[1].plot(range(1, len(classifier2.losses_) + 1), classifier2.losses_, marker='o')
#     ax[1].set_xlabel('Epoch')
#     ax[1].set_ylabel('loss')
#     ax[1].set_title(f'Adaline - learning speed {classifier2.eta}')
#     plt.show()

def plot_losses(classifier: Any) -> None:
    plt.plot(range(1, len(classifier.losses_) + 1), classifier.losses_, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning speed {classifier.eta}')
    plt.show()
