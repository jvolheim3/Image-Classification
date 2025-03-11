import numpy as np

def Identity(x):
    y = x.copy()
    return np.asarray(y)

def Sigmoid(x):
    y = x.copy()
    y = np.asarray(y)
    return 1 / (1 + np.exp(-y))


def Sinusoidal(x):
    y = x.copy()
    y = np.asarray(y)
    return np.sin(y)


def ReLu(x):
    y = x.copy()
    y = np.asarray(y)
    y[y < 0] = 0
    return y

