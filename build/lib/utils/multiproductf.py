from itertools import combinations
import matplotlib.pyplot as plt
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

def mpfcoeffs_m(ks, pf_order, m, print_mode=False):
    try:
        if len(ks) == 0:
            raise ValueError("ks must be a vector of length > 0")
    except TypeError:
        raise ValueError("ks must be a vector of length > 0")
    
    a = cp.Variable(len(ks))
    cost = cp.norm(a, 1)
    constraint_powers = np.zeros(m)
    constraint_powers[1::] = pf_order + 2*np.arange(m - 1)
    constraint_matrix = np.tile(ks, (len(constraint_powers), 1))
    constraint_matrix = constraint_matrix ** (-constraint_powers.reshape((-1, 1)))
    constraints_values = np.zeros(m)
    constraints_values[0] = 1

    problem = cp.Problem(cp.Minimize(cost), [constraint_matrix @ a == constraints_values])
    problem.solve()

    if print_mode:
        print("\n The optimal coefficients are: ")
        for i in range(len(ks)):
            print(f"a[{i}] = {a.value[i]}")
        print(f"The condition number is: {problem.value}")

    return a.value, problem.value


def mpfcoeffs_sergiy(ks, condition_number, pf_order, print=False):
    a = cp.Variable(len(ks))
    ks = np.array(ks, dtype=int)
    cost = cp.norm(a/ks**(2*pf_order), 1)
    constraint_powers = np.zeros(len(ks))
    # constraint_powers[1::] = np.arange(pf_order, 2*pf_order)
    constraint_powers[1::] = pf_order + 2*np.arange(len(ks) - 1)
    constraint_matrix = np.tile(ks, (len(constraint_powers), 1))
    constraint_matrix = constraint_matrix ** (-constraint_powers.reshape((-1, 1)))
    constraints_values = np.zeros(len(ks))
    constraints_values[0] = 1
    problem = cp.Problem(cp.Minimize(cost), [constraint_matrix @ a == constraints_values, cp.norm(a, 1) <= condition_number])
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
    for i, ks in enumerate(k_candidates):
        coefficients[i], condition_numbers[i] = mpfcoeffs(ks, pf_order)
    
    # Sort by condition number
    sorted_indices = np.argsort(condition_numbers)
    coefficients = coefficients[sorted_indices, :]
    condition_numbers = condition_numbers[sorted_indices]
    k_candidates = k_candidates[sorted_indices, :]

    return k_candidates, coefficients, condition_numbers

def search_mpf_coefficients_sergiy(min_k, max_k, nks, pf_order, trials_limit=None, picker_function=lambda arr, n: np.random.choice(arr, n, replace=False)):
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
    for i, ks in enumerate(k_candidates):
        coefficients[i], condition_numbers[i] = mpfcoeffs_sergiy(ks, pf_order)
    
    # Sort by condition number
    sorted_indices = np.argsort(condition_numbers)
    coefficients = coefficients[sorted_indices, :]
    condition_numbers = condition_numbers[sorted_indices]
    k_candidates = k_candidates[sorted_indices, :]

    return k_candidates, coefficients, condition_numbers

def search_mpf_coefficients_m(min_k, max_k, nks, pf_order, m, trials_limit=None, picker_function=lambda arr, n: np.random.choice(arr, n, replace=False)):
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
    for i, ks in enumerate(k_candidates):
        coefficients[i], condition_numbers[i] = mpfcoeffs_m(ks, pf_order, m)
    
    # Sort by condition number
    sorted_indices = np.argsort(condition_numbers)
    coefficients = coefficients[sorted_indices, :]
    condition_numbers = condition_numbers[sorted_indices]
    k_candidates = k_candidates[sorted_indices, :]

    return k_candidates, coefficients, condition_numbers

def a_onenorm_cost(ks, coefficients):
    if len(ks) != len(coefficients):
        raise ValueError("ks and coefficients must have the same length")
    return np.sum(np.abs(coefficients))

def k_a_onenorm_cost(ks, coefficients):
    if len(ks) != len(coefficients):
        raise ValueError("ks and coefficients must have the same length")
    return np.sum(np.abs(ks))*np.sum(np.abs(coefficients))

def k_a_disp_onenorm_cost(ks, coefficients):
    if len(ks) != len(coefficients):
        raise ValueError("ks and coefficients must have the same length")
    return np.sum(np.abs(ks))*np.sum(np.abs(coefficients))*np.sqrt(np.sum((coefficients - np.mean(coefficients))**2))

def search_mpf_coefficients_ccost(k_min, k_max, nks, pf_order, cost_func, trials_limit=None, return_costs=False):
    if k_min > k_max:
        raise TypeError("k_max must be greater than ceil(t)")
    
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

def search_mpf_coefficients_anorm_max(k_min, anorm_max, nks, pf_order, cost_func=a_onenorm_cost):
    if k_min < 1:
        k_min = 1
    anorm = anorm_max + 1
    current_max_k = k_min + nks
    while anorm > anorm_max:
        ks_cands, coeffs_cands, anorm_cands = search_mpf_coefficients_ccost(k_min, current_max_k, nks, pf_order, cost_func)
        ks = ks_cands[0]
        coeffs = coeffs_cands[0]
        anorm = anorm_cands[0]
        current_max_k = current_max_k + 1
    return ks, coeffs, anorm

def search_mpf_coefficients_anorm_max_kmax(k_max, anorm_max, nks, pf_order, cost_func=a_onenorm_cost):
    if k_max <= nks:
        k_max = nks + 1
    anorm = anorm_max + 1
    current_min_k = k_max - nks
    while anorm > anorm_max:
        ks_cands, coeffs_cands, anorm_cands = search_mpf_coefficients_ccost(current_min_k, k_max, nks, pf_order, cost_func)
        ks = ks_cands[0]
        coeffs = coeffs_cands[0]
        anorm = anorm_cands[0]
        current_min_k = current_min_k - 1
    return ks, coeffs, anorm

def search_mpf_coefficients_sergiy_ccost(k_min, k_max, nks, pf_order, cost_func, trials_limit=None, return_costs=False):
    if k_min > k_max:
        raise TypeError("k_max must be greater than ceil(t)")
    
    k_candidates, coefficients, condition_numbers = search_mpf_coefficients_sergiy(k_min, k_max, nks, pf_order, trials_limit)
    cost_func_values = np.array([cost_func(k_candidates[i], coefficients[i]) for i in range(len(k_candidates))])
    resorted_indices = np.argsort(cost_func_values)
    k_candidates = k_candidates[resorted_indices, :]
    coefficients = coefficients[resorted_indices, :]
    condition_numbers = condition_numbers[resorted_indices]
    if return_costs:
        return k_candidates, coefficients, condition_numbers, cost_func_values[resorted_indices]
    else:
        return k_candidates, coefficients, condition_numbers

def search_mpf_coefficients_ccost_m(k_min, k_max, nks, pf_order, m, cost_func, trials_limit=None, return_costs=False):
    if k_min > k_max:
        raise TypeError("k_max must be greater than ceil(t)")
    
    k_candidates, coefficients, condition_numbers = search_mpf_coefficients_m(k_min, k_max, nks, pf_order, m, trials_limit)
    cost_func_values = np.array([cost_func(k_candidates[i], coefficients[i]) for i in range(len(k_candidates))])
    resorted_indices = np.argsort(cost_func_values)
    k_candidates = k_candidates[resorted_indices, :]
    coefficients = coefficients[resorted_indices, :]
    condition_numbers = condition_numbers[resorted_indices]
    if return_costs:
        return k_candidates, coefficients, condition_numbers, cost_func_values[resorted_indices]
    else:
        return k_candidates, coefficients, condition_numbers
    
def mpf_error(t, ks, coeffs, pf_order, alpha_comm):
    return 2*(t**(pf_order+1)*alpha_comm)**2*np.sum(np.abs(coeffs)/ks**(2*pf_order))

def mpf_min_k_improvement(t_arr, alpha_comm, pf_order):
    return np.ceil(t_arr**(1+1/pf_order)*alpha_comm**(1/pf_order)).astype(int)

def plot_mpf_min_k_improvement(t_arr, alpha_comm, pf_order):
    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")
    cmap = plt.get_cmap("Set2")
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.plot(t_arr, mpf_min_k_improvement(t_arr, alpha_comm, pf_order), "o--", markersize=10, markeredgecolor="black", color=cmap(0), label='MPF min k improvement')
    plt.xlabel(r"$t$")
    plt.ylabel(r"$k_{min}$")
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    plt.show()
    plt.rcdefaults()

def get_ks_coeffs_improvement(t_arr, alpha_comm, nks, anorm_max, pf_order):
    min_ks = mpf_min_k_improvement(t_arr, alpha_comm, pf_order)
    ks_improvement = np.zeros((len(t_arr), nks), dtype=int)
    coeffs_improvement = np.zeros((len(t_arr), nks))
    for i, t in enumerate(t_arr):
        ks_improvement[i, :], coeffs_improvement[i, :], this_costs = search_mpf_coefficients_anorm_max(min_ks[i], anorm_max, nks, pf_order, k_a_onenorm_cost)
    return ks_improvement, coeffs_improvement

def get_ks_maxk_dt(t_arr, max_k_dt, nks, anorm_max, pf_order):
    ks_matrix = np.zeros((len(t_arr), nks), dtype=int)
    coeffs_matrix = np.zeros((len(t_arr), nks))
    for i, t in enumerate(t_arr):
        k_max = int(np.ceil(t_arr[i] / max_k_dt))
        ks_matrix[i, :], coeffs_matrix[i, :], this_costs = search_mpf_coefficients_anorm_max_kmax(k_max, anorm_max, nks, pf_order, k_a_onenorm_cost)
    return ks_matrix, coeffs_matrix

def plot_mpf_ks(ks, x_arr=None, x_label=None, y_label=None, nticks=5, filepath=""):
    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")
    cmap = plt.get_cmap("Set2")
    fig, ax = plt.subplots(figsize=(9, 6))
    if x_arr is not None:
        if len(x_arr) != ks.shape[0]:
            raise ValueError("x_arr must have the same length as the number of rowss in ks.")
    else:
        x_arr = np.arange(ks.shape[0])
    x_ticks = np.linspace(0, x_arr[-1], nticks)
    for i in range(ks.shape[1]):
        plt.plot(x_arr, ks[:, i], "o--", markersize=10, markeredgecolor="black", color=cmap(i % 8), label=r"MPF PF_{%i} Layers" % (i + 1))
    plt.xticks(x_ticks)
    plt.xlabel(x_label if x_label is not None else "Sim. index", labelpad=10)
    plt.ylabel("Layers" if y_label is None else y_label, labelpad=10)
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, facecolor="none", dpi=300)
    plt.show()
    plt.rcdefaults()

def plt_mpf_conventional_comparison(x_arr, ks, dt, depth_factor=1, x_label=None, y_label=None, title="", nticks=5, filepath=""):
    if len(x_arr) != ks.shape[0]:
        raise ValueError("x_arr must have the same length as the number of rows in ks.")
    conventional_depth = np.ceil(x_arr / dt).astype(int)

    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")
    cmap = plt.get_cmap("Set2")
    fig, ax = plt.subplots(figsize=(9, 6))
    x_ticks = np.linspace(0, x_arr[-1], nticks)
    for i in range(ks.shape[1]):
        plt.plot(x_arr, depth_factor*ks[:, i], "o--", markersize=10, markeredgecolor="black", color=cmap(i % 8), label=r"MPF $\mathrm{PF}_{%i}$ Layers" % (i + 1))
    plt.plot(x_arr, depth_factor*conventional_depth, "o--", markersize=10, markeredgecolor="black", color=cmap((i+1) % 8), label=r"Conventional $\mathrm{PF}_{%i}$ Layers" % (depth_factor))
    plt.xticks(x_ticks)
    plt.xlabel(x_label if x_label is not None else "Sim. index", labelpad=10)
    plt.ylabel("Layers" if y_label is None else y_label, labelpad=10)
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    if title:
        plt.title(title)
    plt.legend(shadow=True, fontsize=12)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, facecolor="none", dpi=300)
    plt.show()
    plt.rcdefaults()