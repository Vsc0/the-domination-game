import numpy as np


def h(gamma, x) -> np.ndarray:
    assert gamma > 0
    gamma, x = float(gamma), np.asfarray(x)
    return np.clip(x ** gamma, 0, 1)


def inv_h(gamma, x) -> np.ndarray:
    assert gamma > 0
    gamma, x = float(gamma), np.asfarray(x)
    return np.clip(h(1 / gamma, x), 0, 1)


# dilating bubble
def db(df, r, c, x) -> np.ndarray:
    assert(r > 0)
    x = np.atleast_1d(x)
    c = np.atleast_1d(np.squeeze(c))
    assert(1 <= x.ndim <= 3)
    assert(c.ndim == 1)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    elif x.ndim == 3:
        c = c[:, np.newaxis, np.newaxis]
    else:
        c = c[:, np.newaxis]
    assert(len(x) == len(c))
    diff = x - c
    distance = np.linalg.norm(diff, ord=2, axis=0)
    distance[distance == 0] = r
    df_arg = np.minimum(np.ones(distance.shape), np.maximum(np.zeros(distance.shape), distance / r))
    df_out = np.minimum(np.ones(distance.shape), np.maximum(np.zeros(distance.shape), df(df_arg)))
    return np.where((distance > 0) == (distance < r), df_out * diff / distance * r + c, x)
