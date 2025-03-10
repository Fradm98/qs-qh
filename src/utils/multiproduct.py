from itertools import combinations
import cvxpy as cp
import numpy as np

def mpfcoeffs(ks, pf_order, print=False):
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

    if print:
        print("\n The optimal coefficients are: ")
        for i in range(len(ks)):
            print(f"a[{i}] = {a.value[i]}")
        print(f"The condition number is: {problem.value}")

    return a.value, problem.value

def search_mpf_coeffients(min_k, max_k, nks, pf_order, trials_limit=None, picker_function=lambda arr, n: np.random.choice(arr, n, replace=False)):
    if min_k > max_k:
        raise ValueError("min_k must be less than max_k")
    if (not min_k.is_integer()) or (not max_k.is_integer()):
        raise ValueError("min_k and max_k must be integers")
    
    k_candidates = np.array(list(combinations(range(min_k, max_k + 1), nks)))
    if trials_limit is not None:
        k_candidates = picker_function(k_candidates, trials_limit)
    # TODO: Consider multiprocessing
    coefficients = np.zeros((len(k_candidates), nks))
    condition_numbers = np.zeros(len(k_candidates))
    for i, ks in enumerate(k_candidates):
        coefficients[i, :], condition_numbers[i] = mpfcoeffs(ks, pf_order)
    
    # Sort by condition number
    sorted_indices = np.argsort(condition_numbers)
    coefficients = coefficients[sorted_indices, :]
    condition_numbers = condition_numbers[sorted_indices]
    k_candidates = k_candidates[sorted_indices, :]

    return k_candidates, coefficients, condition_numbers
