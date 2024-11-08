import numpy as np

def compute_depolarizing_probability(measured_observable_arr, expected_observable_arr, observable_trace, nqubits):
    # NOTE: Assumes the observable are Pauli Strings. Otherwise an extra term
    #       is needed PRE 104, 035309 (2021)
    try:
        return (1 - np.array(measured_observable_arr))/(np.array(expected_observable_arr) - observable_trace/2**nqubits)
    except ValueError:
        raise ValueError("measured_observable_arr expected_observable_arr and observable_trace must be broadcastable")
    
def renormalize_observable(measured_observable_arr, p_arr, observable_trace, nqubits):
    try:
        return (measured_observable_arr - p_arr*observable_trace/nqubits) / (1 - p_arr)
    except ValueError:
        raise ValueError("measured_observable_arr expected_observable_arr and observable_trace must be broadcastable")