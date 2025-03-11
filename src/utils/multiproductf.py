from multiprocessing import Pool, cpu_count
from itertools import combinations
import cvxpy as cp
import numpy as np

def mpfcoeffs(ks, pf_order, print_mode=False):
    try:
        if len(ks) == 0:
            raise ValueError("ks must be a vector of length > 0")
    except TypeError:
        raise ValueError("ks must be a vector of length > 0")
    
    a = cp.Variable(len(ks))
    cost = cp.norm(a, 1)
    constraint_powers = np.zeros(len(ks))
    constraint_powers[1::] = pf_order + 2*np.arange(len(ks) - 1)
    constraint_matrix = np.tile(ks, (len(constraint_powers), 1))
    constraint_matrix = constraint_matrix ** (-constraint_powers.reshape((-1, 1)))
    constraints_values = np.zeros(len(ks))
    constraints_values[0] = 1
    problem = cp.Problem(cp.Minimize(cost), [constraint_matrix @ a == constraints_values])
    problem.solve()

    if print_mode:
        print("\n The optimal coefficients are: ")
        for i in range(len(ks)):
            print(f"a[{i}] = {a.value[i]}")
        print(f"The condition number is: {problem.value}")

    return a.value, problem.value

def mpfcoeffs_sergiy(ks, pf_order, print=False):
    a = cp.Variable(len(ks))
    ks = np.array(ks, dtype=int)
    cost = cp.norm(a/ks, 1)
    constraint_powers = np.zeros(len(ks))
    constraint_powers[1::] = np.arange(pf_order, 2*pf_order)
    constraint_matrix = np.tile(ks, (len(constraint_powers), 1))
    constraint_matrix = constraint_matrix ** (-constraint_powers.reshape((-1, 1)))
    constraints_values = np.zeros(len(ks))
    constraints_values[0] = 1
    problem = cp.Problem(cp.Minimize(cost), [constraint_matrix @ a == constraints_values])
    problem.solve()

    if print:
        print("\n The optimal coefficients are: ")
        for i in range(len(ks)):
            print(f"a[{i}] = {a.value[i]}")
        print(f"The condition number is: {problem.value}")

    return a.value, problem.value

def search_mpf_coefficients(min_k, max_k, nks, pf_order, trials_limit=None, picker_function=lambda arr, n: np.random.choice(arr, n, replace=False)):
    if min_k < 1:
        raise ValueError("min_k must be greater than 0")
    if min_k > max_k:
        raise ValueError("min_k must be less than max_k")
    if (not min_k.is_integer()) or (not max_k.is_integer()):
        raise ValueError("min_k and max_k must be integers")
    
    k_candidates = np.array(list(combinations(range(min_k, max_k + 1), nks)))
    if trials_limit is not None:
        k_candidates = picker_function(k_candidates, trials_limit)
    coefficients = np.zeros((len(k_candidates), nks))
    condition_numbers = np.zeros(len(k_candidates))
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(mpfcoeffs, [(ks, pf_order) for ks in k_candidates])
        coefficients = np.array([result[0] for result in results])
        condition_numbers = np.array([result[1] for result in results])
    
    # Sort by condition number
    sorted_indices = np.argsort(condition_numbers)
    coefficients = coefficients[sorted_indices, :]
    condition_numbers = condition_numbers[sorted_indices]
    k_candidates = k_candidates[sorted_indices, :]

    return k_candidates, coefficients, condition_numbers

def search_mpf_parameters_ccost(t, nks, pf_order, k_max, cost_func, trials_limit=None, return_costs=False):
    k_min = np.ceil(t).astype(int)
    try:
        if callable(k_max):
            k_max = np.ceil(k_max(t)).astype(int)
        elif k_min > k_max:
            raise TypeError("k_max must be greater than ceil(t)")
    except TypeError:
        raise ValueError("k_max must be a callable or real number")
    
    k_candidates, coefficients, condition_numbers = search_mpf_coefficients(k_min, k_max, nks, pf_order, trials_limit)
    cost_func_values = np.array([cost_func(k_candidates[i], coefficients[i]) for i in range(len(k_candidates))])
    resorted_indices = np.argsort(cost_func_values)
    k_candidates = k_candidates[resorted_indices, :]
    coefficients = coefficients[resorted_indices, :]
    condition_numbers = condition_numbers[resorted_indices]
    if return_costs:
        return k_candidates, coefficients, condition_numbers, cost_func_values[resorted_indices]
    else:
        return k_candidates, coefficients, condition_numbers
    
def k_a_onenorm_cost(ks, coefficients):
    if len(ks) != len(coefficients):
        raise ValueError("ks and coefficients must have the same length")
    return np.sum(np.abs(ks))*np.sum(np.abs(coefficients))