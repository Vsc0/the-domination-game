from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from functools import partial


def pareto_front(problem_name: str,
                 n_obj: int = 2, method_name: str = 'energy', n_dim: int = 2,
                 n_points: int = 1000, n_partitions: int = 24, seed: int = None):

    ref_dirs = None
    if method_name == 'das-dennis':
        ref_dirs = get_reference_directions(name=method_name, n_dim=n_obj, n_partitions=n_partitions)
    elif method_name == 'energy':
        ref_dirs = get_reference_directions(name=method_name, n_dim=n_obj, n_points=n_points)
    else:
        Exception('Method not implemented yet.')

    problem = get_problem(name=problem_name, n_var=n_dim, n_obj=n_obj)

    if problem_name in ('dtlz5', 'dtlz6', 'dtlz7'):
        pf = problem.pareto_front()
    elif problem_name in ('dtlz1', 'dtlz2', 'dtlz3', 'dtlz4'):
        pf = problem.pareto_front(ref_dirs)
    else:
        pf = problem.pareto_front()

    return pf.T


# DTLZ
dtlz1_pf: partial = partial(pareto_front, 'dtlz1')
dtlz2_pf: partial = partial(pareto_front, 'dtlz2')
dtlz3_pf: partial = partial(pareto_front, 'dtlz3')
dtlz4_pf: partial = partial(pareto_front, 'dtlz4')
dtlz5_pf: partial = partial(pareto_front, 'dtlz5')
dtlz6_pf: partial = partial(pareto_front, 'dtlz6')
dtlz7_pf: partial = partial(pareto_front, 'dtlz7')

# WFG
wfg1_pf: partial = partial(pareto_front, 'wfg1')
wfg2_pf: partial = partial(pareto_front, 'wfg2')
wfg3_pf: partial = partial(pareto_front, 'wfg3')
wfg4_pf: partial = partial(pareto_front, 'wfg4')
wfg5_pf: partial = partial(pareto_front, 'wfg5')
wfg6_pf: partial = partial(pareto_front, 'wfg6')
wfg7_pf: partial = partial(pareto_front, 'wfg7')
wfg8_pf: partial = partial(pareto_front, 'wfg8')
wfg9_pf: partial = partial(pareto_front, 'wfg9')