import numpy as np
from functools import partial
from typing import Tuple
from pymoo.problems import get_problem


def problem(problem_name: str, n_obj: int, x) -> Tuple[np.ndarray, ...]:
    x: np.ndarray = np.atleast_1d(x)
    x: np.ndarray = np.array(x, dtype=np.float64, copy=False, order='C')
    x_ndim = x.ndim
    x_shape = x.shape
    if x_ndim == 1:
        x = x[:, np.newaxis]
    elif x_ndim == 3:
        x = np.vstack(x).reshape(x_shape[0], -1)
    n_var = len(x)
    y = get_problem(name=problem_name, n_var=n_var, n_obj=n_obj).evaluate(X=x.T).T
    if x_ndim == 3:
        y = y.reshape((n_obj, *x_shape[1:]))
    return tuple(np.squeeze(z) if z.ndim == 1 else z for z in y)


# DTLZ
dtlz1: partial = partial(problem, 'dtlz1')
dtlz2: partial = partial(problem, 'dtlz2')
dtlz3: partial = partial(problem, 'dtlz3')
dtlz4: partial = partial(problem, 'dtlz4')
dtlz5: partial = partial(problem, 'dtlz5')
dtlz6: partial = partial(problem, 'dtlz6')
dtlz7: partial = partial(problem, 'dtlz7')

# WFG
wfg1: partial = partial(problem, 'wfg1')
wfg2: partial = partial(problem, 'wfg2')
wfg3: partial = partial(problem, 'wfg3')
wfg4: partial = partial(problem, 'wfg4')
wfg5: partial = partial(problem, 'wfg5')
wfg6: partial = partial(problem, 'wfg6')
wfg7: partial = partial(problem, 'wfg7')
wfg8: partial = partial(problem, 'wfg8')
wfg9: partial = partial(problem, 'wfg9')

# Kursawe
kursawe: partial = partial(problem, 'kursawe')
