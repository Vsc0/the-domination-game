import numpy as np


def fonseca(n_obj: int, x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    y1 = 1 - np.exp(-np.sum((x - 1 / np.sqrt(3)) ** 2, axis=0))
    y2 = 1 - np.exp(-np.sum((x + 1 / np.sqrt(3)) ** 2, axis=0))
    if y1.ndim == 1:
        y1 = np.squeeze(y1)
    if y2.ndim == 1:
        y2 = np.squeeze(y2)
    return y1, y2


def kursawe(n_obj: int, x):
    x = np.asfarray(np.atleast_1d(x))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x_i = x[:-1, :]
    x_i_plus_1 = x[1:, :]
    y1 = np.sum(-10 * np.exp(-.2 * np.sqrt(x_i ** 2 + x_i_plus_1 ** 2)), axis=0)
    y2 = np.sum(np.absolute(x) ** .8 + 5 * np.sin(x ** 3), axis=0)
    if y1.ndim == 1:
        y1 = np.squeeze(y1)
    if y2.ndim == 1:
        y2 = np.squeeze(y2)
    return y1, y2
