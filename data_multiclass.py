"""
Generated data for multiclass classification.

Credit: <https://cs231n.github.io/neural-networks-case-study/>
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_spectral_data():
    N = 100  # number of points per class
    D = 2  # dimensionality
    K = 3  # number of classes
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    Y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j+1
    # lets visualize the data:
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    return X, Y
