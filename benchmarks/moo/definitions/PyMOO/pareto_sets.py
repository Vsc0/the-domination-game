import numpy as np
from functools import partial
from pymoo.problems import get_problem


def dtlz1to5_ps(n_obj, n_dim, num):
    x = np.linspace(np.zeros(n_dim), np.ones(n_dim), num).T
    for i in range(n_obj - 1, n_dim):
        x[i, :] = .5
    return x


def dtlz6to7_ps(n_obj, n_dim, num):
    x = np.linspace(np.zeros(n_dim), np.ones(n_dim), num).T
    for i in range(n_obj - 1, n_dim):
        x[i, :] = 0
    return x


def wfg_ps(problem_name, n_dim, n_obj):
    return get_problem(name=problem_name, n_var=n_dim, n_obj=n_obj).pareto_set().T


dtlz1_ps = dtlz1to5_ps
dtlz2_ps: partial = partial(dtlz1to5_ps,)
dtlz3_ps: partial = partial(dtlz1to5_ps,)
dtlz4_ps: partial = partial(dtlz1to5_ps,)
dtlz5_ps: partial = partial(dtlz1to5_ps,)
dtlz6_ps: partial = partial(dtlz6to7_ps,)
dtlz7_ps: partial = partial(dtlz6to7_ps,)

wfg1_ps: partial = partial(wfg_ps, 'wfg1')
wfg2_ps: partial = partial(wfg_ps, 'wfg2')
wfg3_ps: partial = partial(wfg_ps, 'wfg3')
wfg4_ps: partial = partial(wfg_ps, 'wfg4')
wfg5_ps: partial = partial(wfg_ps, 'wfg5')
wfg6_ps: partial = partial(wfg_ps, 'wfg6')
wfg7_ps: partial = partial(wfg_ps, 'wfg7')
wfg8_ps: partial = partial(wfg_ps, 'wfg8')
wfg9_ps: partial = partial(wfg_ps, 'wfg9')
