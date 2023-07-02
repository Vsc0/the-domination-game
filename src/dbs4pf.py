import os
import numpy as np
import array
from pathlib import Path
from pymoo.indicators.hv import HV
from functools import partial
from deap import base, creator, tools
import pickle
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from itertools import combinations

from benchmarks.moo.functions import functions
from benchmarks.moo.limits import limits
from src.df import h, db
from src.df_utils import composite_dilation


def normalize_first_pareto_fronts(a_pf):
    approx_nadir = np.max(a_pf, axis=1)[:, np.newaxis]
    approx_ideal = np.min(a_pf, axis=1)[:, np.newaxis]
    return (a_pf - approx_ideal) / (approx_nadir - approx_ideal)


def select_first_pareto_front(n_obj, function, points):  # use DEAP (https://deap.readthedocs.io/en/master/)
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,) * n_obj)
    creator.create('Individual', array.array, typecode='d', fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register('individual', tools.initIterate, creator.Individual, lambda: [])
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', function)
    # sort non-dominated solutions
    population = toolbox.population(points.shape[1])
    for i, ind in enumerate(population):
        ind.extend(points[:, i])
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  # function evaluations
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    population = tools.emo.sortNondominated(population, len(population), first_front_only=True)[0]
    return population


def sample_points_inside_hypersphere(n_dim, r, c, num, rng):
    lows = np.zeros(num)
    highs = np.ones(num)
    u = rng.uniform(low=lows, high=highs)
    x = rng.normal(loc=lows, scale=highs, size=(n_dim, num))
    s = np.sqrt(np.sum(x ** 2, axis=0))
    x = x / s
    return x * u ** (1 / n_dim) * r + np.expand_dims(c, axis=1)


def perform_experiment_a(func_name, n_obj, n_dim, runs, verbose=False):
    experiment = 'A'
    path = f'results/CEC23/{experiment}/{n_dim}D/{n_obj}O/'
    Path(path).mkdir(parents=True, exist_ok=True)
    output_file = f'{path}{experiment}_{func_name}_{n_dim}D_{n_obj}O_{runs}R.pkl'

    rng = np.random.default_rng()

    if not os.path.isfile(output_file):
        function = partial(functions[func_name], n_obj)
        search_space = limits[func_name][:n_dim] if func_name[:3] == 'WFG' else [limits[func_name]] * n_dim
        lwb, upb = list(zip(*search_space))

        approx_first_pareto_fronts_PF100 = []
        shapes_1_PF100 = []
        approx_first_pareto_fronts_odd = []
        shapes_1_odd = []
        approx_first_pareto_fronts_new = []
        shapes_1_new = []

        for run in range(runs):
            samples = rng.uniform(low=lwb, high=upb, size=(100, n_dim)).T  # PF-100
            odd_samples = samples[:, ::2]  # PF-50

            # PF-100
            pop = select_first_pareto_front(n_obj, function, samples)  # 100 evaluations
            approx_first_pareto_front_PF100 = np.array(tuple(zip(*(ind.fitness.values for ind in pop))))
            approx_first_pareto_fronts_PF100.append(approx_first_pareto_front_PF100)
            shape_1_PF100 = approx_first_pareto_front_PF100.shape[1]
            shapes_1_PF100.append(shape_1_PF100)

            # PF-50
            pop = select_first_pareto_front(n_obj, function, odd_samples)  # 50 evaluations
            approx_first_pareto_set_odd = np.array(tuple(zip(*(ind for ind in pop))))
            approx_first_pareto_front_odd = np.array(tuple(zip(*(ind.fitness.values for ind in pop))))
            approx_first_pareto_fronts_odd.append(approx_first_pareto_front_odd)
            shape_1_odd = approx_first_pareto_front_odd.shape[1]
            shapes_1_odd.append(shape_1_odd)
            # get adjacent solutions in the objective space
            indices = np.argsort(approx_first_pareto_front_odd[0, :], axis=0)
            ij_ps = np.row_stack((indices,) * len(approx_first_pareto_set_odd))
            approx_first_pareto_set_odd = np.take_along_axis(arr=approx_first_pareto_set_odd, indices=ij_ps, axis=1)

            # PF-LBDF
            local_dilations = []
            radii = []
            centers = []
            for i in range(shape_1_odd - 1):
                a = approx_first_pareto_set_odd[:, i]
                b = approx_first_pareto_set_odd[:, i + 1]
                ab_2 = np.linalg.norm(a - b, ord=2)
                radius = ab_2 / 2
                radii.append(radius)
                unit_vector = (a - b) / ab_2
                center = unit_vector * radius + b
                centers.append(center)
                dilation_function = partial(h, 2)
                local_dilation = partial(db, dilation_function, radius, center)
                local_dilations.append(local_dilation)
            dilation = composite_dilation(local_dilations)
            n_lbdf = len(local_dilations)
            if verbose:
                print(f'{func_name} {n_dim}D {n_obj}O {run}R {n_lbdf}Bs')
            # generate 50 new random candidate solutions in the search space regions manipulated by LBDFs
            n_samples = 50
            samples_in_bubbles = np.empty(shape=(n_dim, n_samples))
            for i in range(n_samples):
                if n_lbdf == 0:  # if the first pareto set contains only one solution
                    samples_in_bubbles[:, i:(i + 1)] = rng.uniform(low=lwb, high=upb, size=(1, n_dim)).T
                    continue
                j = rng.choice(a=range(n_lbdf), size=1, replace=True, shuffle=True)
                radius = radii[j[0]]
                center = centers[j[0]]
                xs = sample_points_inside_hypersphere(n_dim, r=radius, c=center, num=1, rng=rng)
                # samples may lie outside the search space
                xs = np.clip(xs, np.expand_dims(lwb, axis=1), np.expand_dims(upb, axis=1))
                samples_in_bubbles[:, i:(i + 1)] = xs
            ld_samples = dilation(samples_in_bubbles)
            # merge with PF-50 and calculate the resulting Pareto front
            pop = select_first_pareto_front(
                n_obj, function, points=np.column_stack((odd_samples, ld_samples)))  # 50 + 50 function evaluations
            approx_first_pareto_front_new = np.array(tuple(zip(*(ind.fitness.values for ind in pop))))
            approx_first_pareto_fronts_new.append(approx_first_pareto_front_new)
            shape_1_new = approx_first_pareto_front_new.shape[1]
            shapes_1_new.append(shape_1_new)

        # hypervolume
        ref_point = np.array((1,) * n_obj)
        norm_hv = HV(ref_point=ref_point)

        normalized = normalize_first_pareto_fronts(np.column_stack(
            approx_first_pareto_fronts_PF100 + approx_first_pareto_fronts_odd + approx_first_pareto_fronts_new))

        shapes_1_PF100_sum = 0
        shapes_1_odd_sum = 0
        shapes_1_new_sum = 0
        shapes_1_100_last_index = sum(shapes_1_PF100)
        shapes_1_odd_last_index = sum(shapes_1_odd)

        data = []
        for run in range(runs):
            # PF-100
            shape_1_PF100 = shapes_1_PF100[run]
            i = shapes_1_PF100_sum
            j = shapes_1_PF100_sum + shape_1_PF100
            shapes_1_PF100_sum += shape_1_PF100
            norm_approx_first_pareto_front_PF100 = normalized[:, i:j]

            # PF-50
            shape_1_odd = shapes_1_odd[run]
            i = shapes_1_odd_sum + shapes_1_100_last_index
            j = shapes_1_odd_sum + shape_1_odd + shapes_1_100_last_index
            shapes_1_odd_sum += shape_1_odd
            norm_approx_first_pareto_front_odd = normalized[:, i:j]

            # PF-LBDF
            shape_1_new = shapes_1_new[run]
            i = shapes_1_new_sum + shapes_1_100_last_index + shapes_1_odd_last_index
            j = shapes_1_new_sum + shape_1_new + shapes_1_100_last_index + shapes_1_odd_last_index
            shapes_1_new_sum += shape_1_new
            norm_approx_first_pareto_front_new = normalized[:, i:j]

            hv_norm_approx_first_pareto_front_PF100 = norm_hv(norm_approx_first_pareto_front_PF100.T)
            hv_norm_approx_first_pareto_front_odd = norm_hv(norm_approx_first_pareto_front_odd.T)
            hv_norm_approx_first_pareto_front_new = norm_hv(norm_approx_first_pareto_front_new.T)

            record_100 = {'func_name': func_name, 'n_obj': n_obj, 'n_dim': n_dim, 'run': run,
                          'hv': hv_norm_approx_first_pareto_front_PF100, 'sample_type': 'PF-100', 'id': 0}
            record_odd = {'func_name': func_name, 'n_obj': n_obj, 'n_dim': n_dim, 'run': run,
                          'hv': hv_norm_approx_first_pareto_front_odd, 'sample_type': 'PF-50', 'id': 1}
            record_new = {'func_name': func_name, 'n_obj': n_obj, 'n_dim': n_dim, 'run': run,
                          'hv': hv_norm_approx_first_pareto_front_new, 'sample_type': 'PF-LBDF', 'id': 2}
            data.extend([record_100, record_odd, record_new])
            if verbose:
                print(f'{func_name} {n_dim}D {n_obj}O {run}R '
                      f'{hv_norm_approx_first_pareto_front_PF100} HV PF-100 '
                      f'{hv_norm_approx_first_pareto_front_odd} HV PF-50 '
                      f'{hv_norm_approx_first_pareto_front_new} HV PF-LBDF')

        with open(output_file, 'wb') as o_f:
            pickle.dump(data, o_f, 5)


def optimize_with_nsga2(func_name, n_dim, n_obj, pop_size, n_gen, seed):
    problem = get_problem(name=func_name, n_var=n_dim, n_obj=n_obj)
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(problem=problem, algorithm=algorithm, termination=('n_gen', n_gen), seed=seed, verbose=False)
    return res.X.T, res.F.T


def optimize_with_nsga3(func_name, n_dim, n_obj, pop_size, n_gen, seed):
    # ref_dirs = get_reference_directions(name='das-dennis', n_dim=n_obj, n_partitions=12)
    ref_dirs = get_reference_directions(name='energy', n_dim=n_obj, n_points=50)
    problem = get_problem(name=func_name, n_var=n_dim, n_obj=n_obj)
    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    res = minimize(problem=problem, algorithm=algorithm, termination=('n_gen', n_gen), seed=seed, verbose=False)
    return res.X.T, res.F.T


def perform_experiment_b(func_name, n_obj, n_dim, runs, algorithm_name='NSGA2', pop_size=50, n_gen=100, verbose=False):
    experiment = 'B'
    path = f'results/CEC23/{experiment}/{n_dim}D/{n_obj}O/'
    Path(path).mkdir(parents=True, exist_ok=True)
    output_file = f'{path}{experiment}_{algorithm_name}_{func_name}_{n_dim}D_{n_obj}O_{runs}R.pkl'

    rng = np.random.default_rng()

    if not os.path.isfile(output_file):
        function = partial(functions[func_name], n_obj)
        search_space = limits[func_name][:n_dim] if func_name[:3] == 'WFG' else [limits[func_name]] * n_dim
        lwb, upb = list(zip(*search_space))

        approx_first_pareto_fronts_100 = []
        shapes_1_100 = []
        approx_first_pareto_fronts_95 = []
        shapes_1_95 = []
        approx_first_pareto_fronts_new = []
        shapes_1_new = []

        for run in range(runs):
            optimizer = None
            if algorithm_name == 'NSGA2':
                optimizer = optimize_with_nsga2
            elif algorithm_name == 'NSGA3':
                optimizer = optimize_with_nsga3

            # 50 pop_size, 100 n_gen
            approx_first_pareto_set_100, approx_first_pareto_front_100 = optimizer(
                func_name=func_name, n_dim=n_dim, n_obj=n_obj, pop_size=pop_size, n_gen=n_gen, seed=None)
            approx_first_pareto_fronts_100.append(approx_first_pareto_front_100)
            shape_1_100 = approx_first_pareto_front_100.shape[1]
            shapes_1_100.append(shape_1_100)

            # 50 pop_size, 95 n_gen
            approx_first_pareto_set_95, approx_first_pareto_front_95 = optimizer(
                func_name=func_name, n_dim=n_dim, n_obj=n_obj, pop_size=pop_size, n_gen=n_gen - 5, seed=None)
            approx_first_pareto_fronts_95.append(approx_first_pareto_front_95)
            shape_1_95 = approx_first_pareto_front_95.shape[1]
            shapes_1_95.append(shape_1_95)

            # LBDF-250
            # get adjacent solutions in the objective space
            indices = np.argsort(approx_first_pareto_front_95[0, :], axis=0)
            ij_ps = np.row_stack((indices,) * len(approx_first_pareto_set_95))
            approx_first_pareto_set_95 = np.take_along_axis(
                arr=approx_first_pareto_set_95, indices=ij_ps, axis=1)
            distances = {}
            pf_combinations = list(combinations(range(shape_1_95), 2))
            for ia, ib in pf_combinations:
                # compute distances in the objective space
                a = approx_first_pareto_front_95[:, ia]
                b = approx_first_pareto_front_95[:, ib]
                ab_2 = np.linalg.norm(a - b, ord=2)
                distances.update({(ia, ib): ab_2})
            indexes = []
            local_dilations = []
            radii = []
            centers = []
            for i in range(shape_1_95 - 1):
                indexes_of_distances_from_i = tuple(filter(lambda x: i in x, distances.keys()))
                distances_from_i = tuple(distances[y] for y in indexes_of_distances_from_i)
                best = np.argmin(distances_from_i)
                index_of_best_distance_from_i = indexes_of_distances_from_i[int(best)]
                if index_of_best_distance_from_i in indexes:
                    continue
                else:
                    indexes.append(index_of_best_distance_from_i)
                    # compute distances in the search space
                    ia, ib = index_of_best_distance_from_i
                    a = approx_first_pareto_set_95[:, ia]
                    b = approx_first_pareto_set_95[:, ib]
                    ab_2 = np.linalg.norm(a - b, ord=2)
                    radius = ab_2 / 2
                    radii.append(radius)
                    unit_vector = (a - b) / ab_2
                    center = unit_vector * radius + b
                    centers.append(center)
                dilation_function = partial(h, 2)
                local_dilation = partial(db, dilation_function, radius, center)
                local_dilations.append(local_dilation)
            dilation = composite_dilation(local_dilations)
            n_lbdf = len(local_dilations)
            if verbose:
                print(f'{func_name} {n_dim}D {n_obj}O {run}R {n_lbdf}Bs')
            # generate 250 new random candidate solutions in the search space regions manipulated by LBDFs
            n_samples = 5 * pop_size
            samples_in_bubbles = np.empty(shape=(n_dim, n_samples))
            for i in range(pop_size):
                if shape_1_95 == 1:
                    samples_in_bubbles[:, i * 5:(i + 1) * 5] = rng.uniform(low=lwb, high=upb, size=(5, n_dim)).T
                    continue
                j = rng.choice(a=range(n_lbdf), size=1, replace=True, shuffle=True)
                radius = radii[j[0]]
                center = centers[j[0]]
                xs = sample_points_inside_hypersphere(n_dim, r=radius, c=center, num=5, rng=rng)
                # samples may lie outside the search space
                xs = np.clip(xs, np.expand_dims(lwb, axis=1), np.expand_dims(upb, axis=1))
                samples_in_bubbles[:, i * 5:(i + 1) * 5] = xs
            ld_samples = dilation(samples_in_bubbles)
            # merge with test NSGAX-4750 and calculate the resulting Pareto front
            pop = select_first_pareto_front(
                n_obj, function, points=np.column_stack((approx_first_pareto_set_95, ld_samples)))
            approx_first_pareto_front_new = np.array(tuple(zip(*(ind.fitness.values for ind in pop))))
            approx_first_pareto_fronts_new.append(approx_first_pareto_front_new)
            shape_1_new = approx_first_pareto_front_new.shape[1]
            shapes_1_new.append(shape_1_new)
            if verbose:
                print(f'n_gen {n_gen} shape_1_100 {shape_1_100}'
                      f'n_gen=95 {n_gen - 5} {shape_1_95}'
                      f'n_gen=5 {shape_1_new}')

        # hypervolume
        ref_point = np.array((1,) * n_obj)
        norm_hv = HV(ref_point=ref_point)

        normalized = normalize_first_pareto_fronts(np.column_stack(
            approx_first_pareto_fronts_100 + approx_first_pareto_fronts_95 + approx_first_pareto_fronts_new))

        shapes_1_100_sum = 0
        shapes_1_95_sum = 0
        shapes_1_new_sum = 0
        shapes_1_100_last_index = sum(shapes_1_100)
        shapes_1_95_last_index = sum(shapes_1_95)

        data = []
        for run in range(runs):
            # 50 pop_size, 100 n_gen
            shape_1_100 = shapes_1_100[run]
            i = shapes_1_100_sum
            j = shapes_1_100_sum + shape_1_100
            shapes_1_100_sum += shape_1_100
            norm_approx_first_pareto_front_100 = normalized[:, i:j]

            # 50 pop_size, 95 n_gen
            shape_1_95 = shapes_1_95[run]
            i = shapes_1_95_sum + shapes_1_100_last_index
            j = shapes_1_95_sum + shape_1_95 + shapes_1_100_last_index
            shapes_1_95_sum += shape_1_95
            norm_approx_first_pareto_front_95 = normalized[:, i:j]

            # LBDF-250
            shape_1_new = shapes_1_new[run]
            i = shapes_1_new_sum + shapes_1_100_last_index + shapes_1_95_last_index
            j = shapes_1_new_sum + shape_1_new + shapes_1_100_last_index + shapes_1_95_last_index
            shapes_1_new_sum += shape_1_new
            norm_approx_first_pareto_front_new = normalized[:, i:j]

            hv_norm_approx_first_pareto_front_100 = norm_hv(norm_approx_first_pareto_front_100.T)
            hv_norm_approx_first_pareto_front_95 = norm_hv(norm_approx_first_pareto_front_95.T)
            hv_norm_approx_first_pareto_front_new = norm_hv(norm_approx_first_pareto_front_new.T)

            record_100 = {'func_name': func_name, 'n_obj': n_obj, 'n_dim': n_dim, 'run': run,
                          'hv': hv_norm_approx_first_pareto_front_100, 'sample_type': 'PF-100', 'id': 0}
            record_95 = {'func_name': func_name, 'n_obj': n_obj, 'n_dim': n_dim, 'run': run,
                         'hv': hv_norm_approx_first_pareto_front_95, 'sample_type': 'PF-95', 'id': 1}
            record_new = {'func_name': func_name, 'n_obj': n_obj, 'n_dim': n_dim, 'run': run,
                          'hv': hv_norm_approx_first_pareto_front_new, 'sample_type': 'PF-LBDF', 'id': 2}
            data.extend([record_100, record_95, record_new])
            if verbose:
                print(f'{func_name} {n_dim}D {n_obj}O {run}R '
                      f'{hv_norm_approx_first_pareto_front_100} HV PF-100 '
                      f'{hv_norm_approx_first_pareto_front_95} HV PF-95 '
                      f'{hv_norm_approx_first_pareto_front_new} HV PF-LBDF')

        with open(output_file, 'wb') as o_f:
            pickle.dump(data, o_f, 5)


def main():
    problem_names = tuple([f'DTLZ{i}' for i in range(1, 8)] + [f'WFG{i}' for i in range(1, 10)])
    # A. Two-Objectives Problems
    for problem_name in problem_names:
        perform_experiment_a(func_name=problem_name, n_obj=2, n_dim=30, runs=30)
    # B. Three-Objectives Problems
    for problem_name in problem_names:
        perform_experiment_b(func_name=problem_name, n_obj=3, n_dim=30, runs=30,
                             algorithm_name='NSGA2', pop_size=50, n_gen=100)


if __name__ == '__main__':
    main()
